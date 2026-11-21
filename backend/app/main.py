"""FastAPI app: GET /health, POST /chat, GET /sessions/{session_id}/messages."""

from __future__ import annotations

import logging
import math
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import Settings, load_settings
from .db import Database
from .llm import AllProvidersUnavailable, LLMGateway, parse_simulation_header
from .schemas import ChatMetadata, ChatRequest, ChatResponse, HealthResponse, HistoryResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.database_url)
        db.init()
        app.state.db = db
        app.state.gateway = LLMGateway(settings)
        if app.state.gateway.keys_configured < 2:
            log.warning("LLM_API_KEYS has %d key(s); at least 2 are needed for failover",
                        app.state.gateway.keys_configured)
        log.info("ready: %d key(s), models=%s, cache=%s, failure simulation=%s",
                 app.state.gateway.keys_configured, app.state.gateway.models,
                 settings.cache_enabled, settings.enable_failure_simulation)
        yield

    app = FastAPI(title="Lyftr AI Chatbot", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request):
        db: Database = request.app.state.db
        gateway: LLMGateway = request.app.state.gateway
        db_ok = await run_in_threadpool(db.ping)
        if not db_ok:
            return JSONResponse(status_code=503, content={"detail": "database unavailable"})
        if gateway.keys_configured == 0:
            return JSONResponse(status_code=503, content={"detail": "no API keys configured"})
        return HealthResponse(
            status="ok",
            keys_configured=gateway.keys_configured,
            cache_enabled=gateway.cache_enabled,
            failure_simulation=settings.enable_failure_simulation,
            models=gateway.models,
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        body: ChatRequest,
        request: Request,
        x_simulate_failure: str | None = Header(default=None),
    ):
        started = time.perf_counter()
        db: Database = request.app.state.db
        gateway: LLMGateway = request.app.state.gateway

        session = await run_in_threadpool(db.get_session, body.session_id)
        system_prompt = body.system_prompt or (session or {}).get("system_prompt") or settings.default_system_prompt
        history = []
        if session and settings.max_history_messages:
            history = await run_in_threadpool(db.get_messages, body.session_id, settings.max_history_messages)

        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": body.message})

        simulate = {}
        if settings.enable_failure_simulation and x_simulate_failure:
            simulate = parse_simulation_header(x_simulate_failure, gateway.key_ids)

        try:
            result = await gateway.complete(messages, simulate=simulate)
        except AllProvidersUnavailable:
            # Nothing is written: a failed request never enters the session history.
            return JSONResponse(status_code=503, content={"detail": "all providers unavailable"})

        # ceil: an honest upper bound, so a sub-millisecond cache hit reads as 1 ms rather than 0
        latency_ms = math.ceil((time.perf_counter() - started) * 1000)
        metadata = ChatMetadata(
            resolution_source=result.resolution_source,
            model=result.model,
            latency_ms=latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )
        await run_in_threadpool(
            db.save_exchange,
            body.session_id,
            body.system_prompt,
            body.message,
            result.reply,
            metadata.model_dump(),
        )
        return ChatResponse(session_id=body.session_id, reply=result.reply, metadata=metadata)

    @app.get("/sessions/{session_id}/messages", responses={200: {"model": HistoryResponse}})
    async def session_messages(session_id: str, request: Request):
        db: Database = request.app.state.db
        session = await run_in_threadpool(db.get_session, session_id)
        if session is None:
            return JSONResponse(status_code=404, content={"detail": "session not found"})
        # Built by hand so user messages carry no "metadata" key (assistant messages only).
        messages = await run_in_threadpool(db.get_messages, session_id)
        return {"session_id": session_id, "messages": messages}

    return app


app = create_app()
