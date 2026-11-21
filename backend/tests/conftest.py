from __future__ import annotations

import socket
import threading
import time
import uuid

import pytest
import uvicorn
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.fake_upstream import STATE
from tests.fake_upstream import app as upstream_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def upstream_url():
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(upstream_app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def upstream():
    STATE.reset()
    yield STATE
    STATE.reset()


def make_settings(upstream_url: str, db_path, **overrides) -> Settings:
    values = dict(
        llm_api_keys=["test-key-1", "test-key-2"],
        llm_model="openai/fake-model",
        llm_api_base=upstream_url,
        database_url=f"sqlite:///{db_path}",
        allowed_origins=["http://localhost:3000"],
        enable_failure_simulation=True,
        cache_enabled=True,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture()
def client(upstream_url, upstream, tmp_path):
    app = create_app(make_settings(upstream_url, tmp_path / "app.db"))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def new_session():
    return lambda: f"s-{uuid.uuid4().hex[:12]}"
