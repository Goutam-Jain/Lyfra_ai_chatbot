"""A tiny OpenAI-compatible server used as the LLM provider in tests.

Tests talk to it through the real LiteLLM network path (no mocking inside LiteLLM).
`STATE.fail_keys` makes a given API key answer with an HTTP error, to test real failover.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


@dataclass
class UpstreamState:
    fail_keys: dict[str, int] = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)
    delay_seconds: float = 0.15

    def reset(self) -> None:
        self.fail_keys.clear()
        self.calls.clear()
        self.delay_seconds = 0.15


STATE = UpstreamState()
app = FastAPI()


def _count_tokens(text: str) -> int:
    return max(1, len(text.split()))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    api_key = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    STATE.calls.append({"api_key": api_key, "messages": body["messages"], "model": body["model"]})

    status = STATE.fail_keys.get(api_key)
    if status:
        return JSONResponse(
            status_code=status,
            content={"error": {"message": f"upstream {status}", "type": "server_error", "code": status}},
        )

    await asyncio.sleep(STATE.delay_seconds)
    messages = body["messages"]
    last_user = messages[-1]["content"]
    turns = sum(1 for m in messages if m["role"] == "user")
    reply = f"(turn {turns}) You said: {last_user}"
    prompt_tokens = sum(_count_tokens(m["content"]) + 3 for m in messages)
    completion_tokens = _count_tokens(reply)
    return {
        "id": f"chatcmpl-{len(STATE.calls)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
