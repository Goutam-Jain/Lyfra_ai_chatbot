from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.llm import parse_simulation_header
from app.main import create_app
from tests.conftest import make_settings

META_FIELDS = {
    "resolution_source", "model", "latency_ms", "input_tokens",
    "output_tokens", "key_id", "fallback_attempts",
}


def chat(client, session_id, message, headers=None, **extra):
    return client.post("/chat", json={"session_id": session_id, "message": message, **extra}, headers=headers or {})


# ---------------------------------------------------------------- health
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["keys_configured"] == 2
    assert body["cache_enabled"] is True


def test_health_without_keys_is_503(upstream_url, tmp_path):
    app = create_app(make_settings(upstream_url, tmp_path / "x.db", llm_api_keys=[]))
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 503
    assert "detail" in r.json()


# ---------------------------------------------------------------- chat contract
def test_chat_live_inference_contract(client, new_session):
    sid = new_session()
    r = chat(client, sid, "What is the capital of France?")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert body["reply"]
    meta = body["metadata"]
    assert META_FIELDS <= meta.keys()
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["model"] == "openai/fake-model"
    assert isinstance(meta["latency_ms"], int) and meta["latency_ms"] >= 0
    assert meta["input_tokens"] > 0 and meta["output_tokens"] > 0
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0
    assert "test-key" not in r.text  # never leak the key itself


def test_message_is_trimmed(client, new_session, upstream):
    chat(client, new_session(), "   hello there   ")
    assert upstream.calls[-1]["messages"][-1]["content"] == "hello there"


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "s-1"},
        {"message": "hi"},
        {"session_id": "", "message": "hi"},
        {"session_id": "s-1", "message": ""},
        {"session_id": "s-1", "message": "   \n\t "},
        {"session_id": "x" * 129, "message": "hi"},
        {"session_id": "s-1", "message": "a" * 4001},
        {"session_id": "s-1", "message": 123},
    ],
)
def test_validation_errors_are_422(client, upstream, payload):
    r = client.post("/chat", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()
    assert upstream.calls == []


def test_message_of_exactly_4000_chars_after_trim_is_ok(client, new_session):
    assert chat(client, new_session(), "  " + "a" * 4000 + "  ").status_code == 200


def test_malformed_json_is_422(client):
    r = client.post("/chat", content=b'{"session_id": "s-1", "message": ', headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_failed_validation_not_stored(client):
    client.post("/chat", json={"session_id": "s-bad", "message": ""})
    assert client.get("/sessions/s-bad/messages").status_code == 404


# ---------------------------------------------------------------- caching
def test_same_first_message_in_new_session_hits_cache(client, new_session, upstream):
    first = chat(client, new_session(), "Tell me a fun fact").json()
    second = chat(client, new_session(), "Tell me a fun fact").json()

    assert first["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert second["metadata"]["resolution_source"] == "LLM_CACHE"
    assert second["reply"] == first["reply"]
    assert second["metadata"]["key_id"] is None
    assert second["metadata"]["fallback_attempts"] == 0
    assert second["metadata"]["input_tokens"] == first["metadata"]["input_tokens"]
    assert second["metadata"]["output_tokens"] == first["metadata"]["output_tokens"]
    assert second["metadata"]["latency_ms"] < first["metadata"]["latency_ms"]
    assert len(upstream.calls) == 1  # the second request never reached the provider


def test_different_message_never_from_cache(client, new_session):
    chat(client, new_session(), "Message one")
    r = chat(client, new_session(), "Message two").json()
    assert r["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_key_includes_history_and_system_prompt(client, new_session):
    chat(client, new_session(), "Hello")
    # same text, but different history -> different message list -> live
    sid = new_session()
    chat(client, sid, "Something else first")
    assert chat(client, sid, "Hello").json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    # same text, different system prompt -> live
    r = chat(client, new_session(), "Hello", system_prompt="You are a pirate.").json()
    assert r["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_disabled(upstream_url, upstream, tmp_path, new_session):
    app = create_app(make_settings(upstream_url, tmp_path / "c.db", cache_enabled=False))
    with TestClient(app) as c:
        assert c.get("/health").json()["cache_enabled"] is False
        chat(c, new_session(), "same")
        assert chat(c, new_session(), "same").json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"


# ---------------------------------------------------------------- tokens
def test_tokens_are_per_request_not_cumulative(client, new_session):
    a = chat(client, new_session(), "alpha beta gamma").json()["metadata"]
    b = chat(client, new_session(), "delta omega sigma").json()["metadata"]
    assert a["input_tokens"] == b["input_tokens"]

    sid = new_session()
    t1 = chat(client, sid, "one two three").json()["metadata"]
    t2 = chat(client, sid, "four").json()["metadata"]
    # second turn's prompt contains the history, but output tokens are only this reply's
    assert t2["input_tokens"] > t1["input_tokens"]
    assert t2["output_tokens"] < t1["input_tokens"] + t1["output_tokens"] + t2["input_tokens"]


# ---------------------------------------------------------------- multi-turn
def test_multi_turn_history_sent_and_stored(client, new_session, upstream):
    sid = new_session()
    chat(client, sid, "My name is Ada.")
    r = chat(client, sid, "What is my name?").json()
    assert r["reply"].startswith("(turn 2)")

    sent = upstream.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1]["content"] == "My name is Ada."

    hist = client.get(f"/sessions/{sid}/messages").json()
    assert hist["session_id"] == sid
    msgs = hist["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "My name is Ada."
    assert "metadata" not in msgs[0]
    assert msgs[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert isinstance(msgs[1]["metadata"]["input_tokens"], int)
    assert all(m["role"] != "system" for m in msgs)


def test_sessions_are_isolated(client, new_session, upstream):
    a, b = new_session(), new_session()
    chat(client, a, "I live in Paris")
    chat(client, b, "Where do I live?")
    assert [m["role"] for m in upstream.calls[-1]["messages"]] == ["system", "user"]
    assert len(client.get(f"/sessions/{b}/messages").json()["messages"]) == 2


def test_unknown_session_404(client):
    r = client.get("/sessions/does-not-exist/messages")
    assert r.status_code == 404
    assert r.json() == {"detail": "session not found"}


def test_system_prompt_persists_for_session(client, new_session, upstream):
    sid = new_session()
    chat(client, sid, "hi", system_prompt="Answer like a pirate.")
    chat(client, sid, "again")
    assert upstream.calls[-1]["messages"][0] == {"role": "system", "content": "Answer like a pirate."}


def test_history_survives_restart(upstream_url, upstream, tmp_path, new_session):
    db = tmp_path / "persist.db"
    sid = new_session()
    with TestClient(create_app(make_settings(upstream_url, db))) as c:
        chat(c, sid, "remember me")
    with TestClient(create_app(make_settings(upstream_url, db))) as c:
        assert len(c.get(f"/sessions/{sid}/messages").json()["messages"]) == 2


# ---------------------------------------------------------------- failover (simulated)
@pytest.mark.parametrize("header", ["key_1=429", "key_1=500", "key_1=502", "key_1=503"])
def test_simulated_failure_fails_over_to_key_2(client, new_session, upstream, header):
    r = chat(client, new_session(), f"failover {header}", headers={"X-Simulate-Failure": header})
    assert r.status_code == 200
    meta = r.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert r.json()["reply"].startswith("(turn 1)")
    assert [c["api_key"] for c in upstream.calls] == ["test-key-2"]  # key_1 never reached the provider


def test_simulation_bypasses_cache_so_failover_is_real(client, new_session):
    chat(client, new_session(), "What is the capital of France?")
    r = chat(client, new_session(), "What is the capital of France?", headers={"X-Simulate-Failure": "key_1=429"})
    assert r.json()["metadata"]["key_id"] == "key_2"
    assert r.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_simulated_all_keys_fail_503_and_nothing_stored(client, new_session, upstream):
    sid = new_session()
    chat(client, sid, "first turn")
    r = chat(client, sid, "this one fails", headers={"X-Simulate-Failure": "*=429"})
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}
    assert len(client.get(f"/sessions/{sid}/messages").json()["messages"]) == 2

    fresh = new_session()
    r = chat(client, fresh, "never stored", headers={"X-Simulate-Failure": "key_1=429,key_2=500"})
    assert r.status_code == 503
    assert client.get(f"/sessions/{fresh}/messages").status_code == 404


def test_simulation_is_scoped_to_one_request(client, new_session):
    chat(client, new_session(), "a", headers={"X-Simulate-Failure": "key_1=429"})
    r = chat(client, new_session(), "b").json()
    assert r["metadata"]["key_id"] == "key_1"


def test_simulation_ignored_when_disabled(upstream_url, upstream, tmp_path, new_session):
    app = create_app(make_settings(upstream_url, tmp_path / "d.db", enable_failure_simulation=False))
    with TestClient(app) as c:
        r = chat(c, new_session(), "hello", headers={"X-Simulate-Failure": "*=429"})
    assert r.status_code == 200
    assert r.json()["metadata"]["key_id"] == "key_1"


# ---------------------------------------------------------------- failover (real upstream errors)
@pytest.mark.parametrize("status", [429, 500, 503])
def test_real_upstream_error_fails_over(client, new_session, upstream, status):
    upstream.fail_keys["test-key-1"] = status
    r = chat(client, new_session(), f"real failure {status}")
    assert r.status_code == 200
    assert r.json()["metadata"]["key_id"] == "key_2"
    assert r.json()["metadata"]["fallback_attempts"] == 1
    # each key is tried exactly once, in order (no hidden SDK retries on the failing key)
    assert [c["api_key"] for c in upstream.calls] == ["test-key-1", "test-key-2"]


def test_real_upstream_all_fail_503(client, new_session, upstream):
    upstream.fail_keys.update({"test-key-1": 429, "test-key-2": 500})
    r = chat(client, new_session(), "everything is down")
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}


# ---------------------------------------------------------------- misc
def test_parse_simulation_header():
    keys = ["key_1", "key_2"]
    assert parse_simulation_header("key_1=429", keys) == {"key_1": 429}
    assert parse_simulation_header(" key_1=429 , key_2=500 ", keys) == {"key_1": 429, "key_2": 500}
    assert parse_simulation_header("*=503", keys) == {"*": 503}
    assert parse_simulation_header("key_9=429,key_1=404,garbage,key_2=abc", keys) == {}
    assert parse_simulation_header(None, keys) == {}


def test_cors_allows_frontend_origin(client):
    r = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-simulate-failure",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_unknown_route_is_json(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.json() == {"detail": "Not Found"}
