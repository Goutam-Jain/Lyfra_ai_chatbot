"""LiteLLM gateway: key pool with ordered failover, response cache and failure simulation.

* Failover - a LiteLLM ``Router`` with one model group per API key (``key_1``, ``key_2``, ...)
  and an ordered fallback chain ``key_1 -> key_2 -> ...``. Retries and cooldowns are off so
  every request walks the chain in the same order and a simulated failure never leaks into
  the next request.
* Caching - LiteLLM's built-in ``Cache`` (in-memory). The key is the LiteLLM model name plus
  the full message list (system prompt + history + new message), nothing else, so the API
  key that answered never splits the cache.
* Simulation - a LiteLLM ``CustomLogger`` whose ``async_pre_call_check`` runs right before
  each deployment is called and raises the provider error requested for that key in the
  current request (read from a ContextVar, so it is scoped to one request only).
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import Router
from litellm.caching.caching import Cache
from litellm.integrations.custom_logger import CustomLogger

from .config import Settings

log = logging.getLogger("app.llm")

LIVE_INFERENCE = "LIVE_INFERENCE"
LLM_CACHE = "LLM_CACHE"
SIMULATABLE_STATUSES = {429, 500, 502, 503}

# key_id -> HTTP status to simulate, for the current request only. "*" matches every key.
_simulated_failures: ContextVar[dict[str, int]] = ContextVar("simulated_failures", default={})

litellm.suppress_debug_info = True
litellm.drop_params = True
# The router logs every failed attempt at ERROR with a long debug dump; we log our own summary.
for _name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)


class AllProvidersUnavailable(Exception):
    """Every key in the pool failed for this request."""


@dataclass(frozen=True)
class KeySlot:
    key_id: str
    model: str
    api_key: str


@dataclass
class CompletionResult:
    reply: str
    resolution_source: str
    model: str
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


def parse_simulation_header(value: str | None, key_ids: list[str]) -> dict[str, int]:
    """Parse ``key_1=429,key_2=500`` / ``*=429``. Unknown keys or statuses are ignored."""
    failures: dict[str, int] = {}
    if not value:
        return failures
    for part in value.split(","):
        if "=" not in part:
            continue
        key, _, status = part.partition("=")
        key, status = key.strip(), status.strip()
        try:
            code = int(status)
        except ValueError:
            continue
        if code not in SIMULATABLE_STATUSES:
            continue
        if key == "*" or key in key_ids:
            failures[key] = code
    return failures


def _simulated_error(status: int, key_id: str, model: str) -> Exception:
    provider = model.split("/", 1)[0] if "/" in model else "openai"
    message = f"Simulated HTTP {status} for {key_id} (X-Simulate-Failure)"
    if status == 429:
        return litellm.RateLimitError(message=message, llm_provider=provider, model=model)
    if status == 502:
        return litellm.BadGatewayError(message=message, llm_provider=provider, model=model)
    if status == 503:
        return litellm.ServiceUnavailableError(message=message, llm_provider=provider, model=model)
    return litellm.InternalServerError(message=message, llm_provider=provider, model=model)


class FailureSimulator(CustomLogger):
    """Fails chosen deployments inside one request, before the provider is called."""

    def _check(self, deployment: dict) -> None:
        failures = _simulated_failures.get()
        if not failures:
            return
        key_id = (deployment.get("model_info") or {}).get("id")
        status = failures.get(key_id) or failures.get("*")
        if status:
            model = (deployment.get("litellm_params") or {}).get("model", "unknown")
            raise _simulated_error(status, key_id, model)

    async def async_pre_call_check(self, deployment: dict, parent_otel_span: Any = None) -> dict | None:
        self._check(deployment)
        return None

    def pre_call_check(self, deployment: dict) -> dict | None:
        self._check(deployment)
        return None


def _install_simulator() -> FailureSimulator:
    for cb in litellm.callbacks:
        if isinstance(cb, FailureSimulator):
            return cb
    sim = FailureSimulator()
    litellm.callbacks.append(sim)
    return sim


class LLMGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.slots = [
            KeySlot(key_id=f"key_{i + 1}", model=settings.model_for_key(i), api_key=key)
            for i, key in enumerate(settings.llm_api_keys)
        ]
        self.key_ids = [s.key_id for s in self.slots]
        self.cache: Cache | None = (
            Cache(type="local", ttl=settings.cache_ttl_seconds) if settings.cache_enabled else None
        )
        self.router: Router | None = self._build_router() if self.slots else None
        _install_simulator()

    # ------------------------------------------------------------------ setup
    def _build_router(self) -> Router:
        model_list = []
        for slot in self.slots:
            params: dict[str, Any] = {"model": slot.model, "api_key": slot.api_key}
            if self.settings.llm_api_base:
                params["api_base"] = self.settings.llm_api_base
            model_list.append(
                {"model_name": slot.key_id, "litellm_params": params, "model_info": {"id": slot.key_id}}
            )
        chain = self.key_ids[1:]
        return Router(
            model_list=model_list,
            fallbacks=[{self.key_ids[0]: chain}] if chain else [],
            num_retries=0,
            disable_cooldowns=True,
            timeout=self.settings.llm_timeout_seconds,
        )

    @property
    def keys_configured(self) -> int:
        return len(self.slots)

    @property
    def cache_enabled(self) -> bool:
        return self.cache is not None

    @property
    def models(self) -> list[str]:
        return list(dict.fromkeys(s.model for s in self.slots))

    # ------------------------------------------------------------------ cache
    async def _cache_lookup(self, messages: list[dict[str, str]]) -> CompletionResult | None:
        if self.cache is None:
            return None
        for model in self.models:
            try:
                hit = await self.cache.async_get_cache(model=model, messages=messages)
            except Exception:  # a broken cache must never break a chat request
                log.exception("cache lookup failed")
                return None
            if isinstance(hit, dict) and hit.get("reply"):
                return CompletionResult(
                    reply=hit["reply"],
                    resolution_source=LLM_CACHE,
                    model=hit.get("model", model),
                    input_tokens=int(hit.get("input_tokens", 0)),
                    output_tokens=int(hit.get("output_tokens", 0)),
                    key_id=None,
                    fallback_attempts=0,
                )
        return None

    async def _cache_store(self, messages: list[dict[str, str]], result: CompletionResult) -> None:
        if self.cache is None:
            return
        entry = {
            "reply": result.reply,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cached_at": time.time(),
        }
        try:
            await self.cache.async_add_cache(entry, model=result.model, messages=messages)
        except Exception:
            log.exception("cache write failed")

    # ------------------------------------------------------------- completion
    async def complete(
        self, messages: list[dict[str, str]], simulate: dict[str, int] | None = None
    ) -> CompletionResult:
        # A request that simulates failures skips the cache lookup: its purpose is to exercise
        # the key pool, and a cache hit would hide the failover entirely.
        if not simulate:
            cached = await self._cache_lookup(messages)
            if cached is not None:
                return cached

        if self.router is None:
            raise AllProvidersUnavailable("no API keys configured")

        token = _simulated_failures.set(dict(simulate or {}))
        try:
            response = await self.router.acompletion(model=self.key_ids[0], messages=messages)
        except Exception as exc:
            log.warning("all %d key(s) failed: %s", len(self.slots), type(exc).__name__)
            raise AllProvidersUnavailable(str(exc)) from exc
        finally:
            _simulated_failures.reset(token)

        hidden = getattr(response, "_hidden_params", {}) or {}
        key_id = hidden.get("model_id") or hidden.get("model_group")
        if key_id not in self.key_ids:
            key_id = self.key_ids[0]
        index = self.key_ids.index(key_id)
        slot = self.slots[index]

        reply = ""
        try:
            reply = (response.choices[0].message.content or "").strip()
        except (AttributeError, IndexError):
            pass
        if not reply:
            log.warning("%s returned an empty reply", key_id)
            raise AllProvidersUnavailable("empty reply")

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        if input_tokens is None:
            input_tokens = litellm.token_counter(model=slot.model, messages=messages)
        if output_tokens is None:
            output_tokens = litellm.token_counter(model=slot.model, text=reply)

        if index:
            log.info("answered by %s after %d failed key(s)", key_id, index)

        result = CompletionResult(
            reply=reply,
            resolution_source=LIVE_INFERENCE,
            model=slot.model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            key_id=key_id,
            fallback_attempts=index,
        )
        await self._cache_store(messages, result)
        return result
