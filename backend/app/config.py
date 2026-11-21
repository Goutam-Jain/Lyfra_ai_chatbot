"""Application settings, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def _float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Key pool. Key ids are key_1, key_2, ... in this order.
    llm_api_keys: list[str] = field(default_factory=list)
    # One model for every key, or a comma-separated list aligned with the keys
    # (lets the pool span several models / providers).
    llm_model: str = DEFAULT_MODEL
    llm_models: list[str] = field(default_factory=list)
    llm_api_base: str | None = None
    llm_timeout_seconds: float = 30.0

    database_url: str = "sqlite:///./data/app.db"
    allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    enable_failure_simulation: bool = False

    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400

    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_history_messages: int = 40

    def model_for_key(self, index: int) -> str:
        if not self.llm_models:
            return self.llm_model
        if index < len(self.llm_models):
            return self.llm_models[index]
        return self.llm_models[-1]


def load_settings() -> Settings:
    env = os.environ
    return Settings(
        llm_api_keys=_split_csv(env.get("LLM_API_KEYS")),
        llm_model=(env.get("LLM_MODEL") or DEFAULT_MODEL).strip(),
        llm_models=_split_csv(env.get("LLM_MODELS")),
        llm_api_base=(env.get("LLM_API_BASE") or "").strip() or None,
        llm_timeout_seconds=_float(env.get("LLM_TIMEOUT_SECONDS"), 30.0),
        database_url=(env.get("DATABASE_URL") or "sqlite:///./data/app.db").strip(),
        allowed_origins=_split_csv(env.get("ALLOWED_ORIGINS")) or ["http://localhost:3000"],
        enable_failure_simulation=_bool(env.get("ENABLE_FAILURE_SIMULATION"), False),
        cache_enabled=_bool(env.get("CACHE_ENABLED"), True),
        cache_ttl_seconds=_int(env.get("CACHE_TTL_SECONDS"), 86400),
        default_system_prompt=(env.get("DEFAULT_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT).strip(),
        max_history_messages=max(0, _int(env.get("MAX_HISTORY_MESSAGES"), 40)),
    )
