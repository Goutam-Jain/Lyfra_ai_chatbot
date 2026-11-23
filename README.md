# Lyftr Chat: LLM chatbot with key failover and response caching

A full-stack chatbot. The **React** frontend talks to a **FastAPI** backend. The backend sends LLM calls through a **LiteLLM** gateway that has a pool of API keys with automatic failover and a response cache. Session history is stored in **SQLite**.

```
┌──────────────┐   POST /chat    ┌──────────────────────────────────────────────┐
│  React (Vite) │ ─────────────▶ │ FastAPI                                       │
│  nginx :3000  │ ◀───────────── │  ├─ SQLite (sessions, messages)  /data/app.db │
└──────────────┘  reply+metadata │  └─ LLMGateway                                │
                                 │      ├─ LiteLLM Cache (model + messages)      │
                                 │      └─ LiteLLM Router                        │
                                 │          key_1 ─▶ key_2 ─▶ … (ordered)        │
                                 └──────────────────────────────────────────────┘
```

---

## 1. Live URLs

| Tier | URL |
|---|---|
| Frontend | `https://lyfra-ai-chatbot.vercel.app` |
| Backend  | `https://lyfra-chat-backend.onrender.com` (health: `https://lyfra-chat-backend.onrender.com/health`) |
| Demo video | `https://drive.google.com/file/d/1I_eP8UtT0slQ1-105V3wRwJy1hUwcRMQ/view?usp=drivesdk` |

---

## 2. Run locally

### With Docker (recommended)

```bash
cp .env.example .env        # then put at least 2 API keys in LLM_API_KEYS
make up                     # builds and starts both services
```

* Frontend: http://localhost:3000
* Backend: http://localhost:8000 (try http://localhost:8000/health)
* `make down` stops the stack. The SQLite file lives in the named volume `chat-data` (mounted at `/data`), so history survives restarts.
* `make logs` follows the logs, and `make test` runs the backend tests.

### Without Docker

Requires Python 3.12+ and Node 20+.

```bash
# backend (terminal 1)
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
export LLM_API_KEYS="key-one,key-two" ENABLE_FAILURE_SIMULATION=true DATABASE_URL=sqlite:///./data/app.db
.venv/bin/uvicorn app.main:app --reload --port 8000

# frontend (terminal 2)
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev      # http://localhost:3000

# tests
cd backend && .venv/bin/python -m pytest
```

`make dev-backend` and `make dev-frontend` do the same and read `.env`.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LLM_API_KEYS` | (required) | Comma-separated key pool, at least 2. Ids are `key_1`, `key_2`, … in this order. |
| `LLM_MODEL` | `groq/openai/gpt-oss-120b` | LiteLLM model name used for every key. |
| `LLM_MODELS` | (empty) | Optional. One model per key, in the same order, so the pool can span several models or providers (e.g. `gemini/gemini-2.5-flash,groq/llama-3.3-70b-versatile`). |
| `LLM_API_BASE` | (empty) | Optional custom base URL (OpenAI-compatible gateways). |
| `LLM_TIMEOUT_SECONDS` | `30` | Timeout per provider call. A timeout also triggers failover. |
| `DATABASE_URL` | `sqlite:////data/app.db` (Docker) | SQLite location. |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins. |
| `ENABLE_FAILURE_SIMULATION` | `false` (`true` in compose) | Enables the `X-Simulate-Failure` header. |
| `CACHE_ENABLED` | `true` | Turns the LiteLLM response cache on or off. |
| `CACHE_TTL_SECONDS` | `86400` | How long a cached response lives. |
| `DEFAULT_SYSTEM_PROMPT` | `You are a helpful, concise assistant.` | Used when a session has no system prompt. |
| `MAX_HISTORY_MESSAGES` | `40` | How many earlier messages are sent as context (bounds cost). |
| `PORT` | `8000` | Port the backend listens on (set automatically by Render, Railway and similar hosts). |
| `API_BASE_URL` (frontend container) | `http://localhost:8000` | Backend URL as the browser sees it. It is written into `env.js` when the container starts. |
| `VITE_API_BASE_URL` (frontend build) | (empty) | Build-time backend URL, for static hosts such as Vercel or Netlify. |

---

## 3. Design decisions

### Key fallback (LiteLLM Router)

* Each API key is its own Router model group (`key_1`, `key_2`, …, with `model_info.id` set to the same id). There is one ordered fallback chain: `key_1 → key_2 → … → key_N`.
* `num_retries=0` and `disable_cooldowns=True`. Every request walks the chain in the same order, so `key_id` and `fallback_attempts` are deterministic: `fallback_attempts` is the position of the answering key in the chain. A failure in one request never changes routing for the next one. The cost is that a key that really is exhausted is tried once per request, which adds one fast failing call before failover. I chose that over hidden shared state.
* Any provider error triggers failover to the next key: 429, quota errors (`RESOURCE_EXHAUSTED`), 5xx, timeouts and connection errors, and also 401/400 "invalid key". One dead key never takes the app down. The tests also check that the provider SDK does not silently retry a failing key, so each key is called exactly once.
* If every key fails, the response is `503 {"detail": "all providers unavailable"}` and nothing is written to SQLite. Validation errors return FastAPI's normal `422` JSON. A catch-all handler turns any unexpected error into JSON with a `detail` field, so the API never returns a stack trace or an HTML page. The API key itself is never returned or logged, only its id.
* **`X-Simulate-Failure`** is built on LiteLLM's own callback system. A `CustomLogger` with `async_pre_call_check` runs right before the Router calls each deployment. It reads the failures requested for the current request from a `ContextVar`, which keeps them scoped to that one request even when requests run concurrently. It then raises the real LiteLLM exception for that status (`RateLimitError`, `InternalServerError`, `BadGatewayError`, `ServiceUnavailableError`). The Router handles the simulated error exactly like a real provider error, and the next key gives a real answer. The header is ignored unless `ENABLE_FAILURE_SIMULATION=true`.
* **Simulated requests skip the cache lookup.** A cache hit would hide the failover you are trying to demonstrate. For example, `key_1=429` on a question that is already cached must still answer from `key_2`. The live answer is still written to the cache.

### Caching (LiteLLM `Cache`, in-memory)

* The cache key is the **LiteLLM model name plus the full message list** (system prompt, history and new message), hashed by `litellm.Cache.get_cache_key`. Because the key pool is not part of the cache key, a reply generated by `key_2` is reused by later requests no matter which key would have answered.
* I call the cache explicitly (`async_get_cache` / `async_add_cache`) instead of passing `caching=True` to the Router. In the LiteLLM version used here (1.110), Router-level caching never produced hits in my tests, and LiteLLM's default key also includes the API key, which would split the cache per key. Calling it explicitly makes the key exactly what the spec asks for and makes `LLM_CACHE` versus `LIVE_INFERENCE` reliable.
* Each cache entry stores the reply, the model and the **token usage of the original call**. A cache hit therefore returns the identical text and the original tokens, with `key_id: null` and `fallback_attempts: 0`. A hit is an in-process dictionary lookup (about 1 ms), much faster than a live call.
* The cache is in memory, so it is lost when the process restarts. That is fine for a cache. Redis-backed caching was out of scope because the spec allows no external services.

### SQLite session storage

* Two tables are created on startup. `messages` follows the required schema (role, content, JSON `metadata` on assistant rows, ISO-8601 `created_at`) with an index on `(session_id, id)`. `sessions` stores the session's own system prompt.
* A turn is written **after** the LLM succeeds: the session upsert, the user row and the assistant row are saved in one transaction. A failed request (422 or 503) leaves no trace, and history can never contain a question without an answer.
* The client sends only the new message. The backend rebuilds the context as system prompt, the last `MAX_HISTORY_MESSAGES` messages and the new message.
* `GET /sessions/{id}/messages` returns the history oldest first, without the system prompt. It returns `404 {"detail":"session not found"}` for an id that has never had a successful turn.
* WAL journal mode and a short-lived connection per operation (run in a threadpool) keep the async event loop free. `DATABASE_URL` points into the `/data` Docker volume.

### Tokens and latency

* `input_tokens` and `output_tokens` come from the provider's `usage` for this one request; they are never a running total. If a provider omits usage, LiteLLM's `token_counter` estimates it.
* `latency_ms` is measured on the server from when the request arrives until the response is ready, rounded up to whole milliseconds.

---

## 4. Verification

Set `B` to your backend URL first, for example:

```bash
B=http://localhost:8000
```

**Health**

```bash
curl -s $B/health
# {"status":"ok","keys_configured":2,"cache_enabled":true,...}
```

**Cache hit.** Send the same first message in two new sessions:

```bash
curl -s -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-1","message":"What is the capital of France?"}'
# resolution_source: LIVE_INFERENCE, key_id: key_1, latency ~ hundreds of ms

curl -s -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-2","message":"What is the capital of France?"}'
# resolution_source: LLM_CACHE, identical reply, same tokens, key_id: null, latency ~1 ms
```

**Failover**

```bash
curl -s -X POST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: key_1=429' -d '{"session_id":"s-3","message":"Hi!"}'
# 200, key_id: key_2, fallback_attempts: 1

curl -s -X POST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: key_1=500' -d '{"session_id":"s-3","message":"And again"}'
# 200, key_id: key_2

curl -s -i -X POST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: *=429' -d '{"session_id":"s-3","message":"fail"}'
# 503 {"detail":"all providers unavailable"} and the message is NOT added to s-3's history
```

**Multi-turn and history**

```bash
curl -s -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-4","message":"My name is Ada."}'
curl -s -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-4","message":"What is my name?"}'   # remembers "Ada"
curl -s $B/sessions/s-4/messages
curl -s $B/sessions/unknown/messages      # 404 {"detail":"session not found"}
```

**Validation**

```bash
curl -s -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-5","message":"   "}'   # 422
curl -s -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":'                        # 422 malformed JSON
```

**In the UI**

* Every assistant reply shows its source (green `LLM_CACHE` or blue `LIVE_INFERENCE`), model, latency, input and output tokens for that request only, the key that answered, and a "failover × N" tag when keys failed first.
* **Settings → Simulate failure** sends `X-Simulate-Failure` with your next messages (presets: `key_1=429`, `key_1=500`, `*=429`, or a custom value), so failover and the 503 message can be shown without curl.
* Click **New chat** (or reload the page) and ask the same first question again to see a cache hit.
* Errors (422, 503, backend unreachable) appear as a red message in the chat with a Retry button.

**Automated tests.** `cd backend && .venv/bin/python -m pytest` runs 40 tests. They use a local fake OpenAI-compatible provider, so they go through the real LiteLLM HTTP path without needing API keys. They cover the API contract, 422 cases, cache hit and miss rules, per-request tokens, multi-turn and session isolation, persistence across restarts, simulated and real upstream failover (429/500/502/503), the 503 path and its "nothing stored" rule, and CORS.

---

## 5. Setup Used

* **Editor:** `VS Code`
* **AI tools:** Claude
* **Stack:** Python 3.12, FastAPI, LiteLLM 1.110, SQLite, React 19 with Vite and TypeScript, nginx, Docker Compose.

---

## 6. Bonus features

* **Failure simulator in the UI.** A dropdown and free-text field send `X-Simulate-Failure`, so failover can be demonstrated in the browser.
* **`key_id` and failover count in the UI**, next to the required metadata, so you can see which key answered.
* **Real-error failover test coverage.** Besides the simulated header, the tests make the fake provider return real HTTP 429/500/503 for one key.
* **Multi-model pools.** `LLM_MODELS` assigns a different model or provider to each key, and the cache is keyed per model.
* **Per-session system prompt** (`system-prompt-input`). It is saved with the session and reused on later turns until it is changed.
* **Bounded context.** `MAX_HISTORY_MESSAGES` caps how much history is sent, which limits token cost on long chats.
* **Runtime frontend config.** The frontend Docker image writes `env.js` from `API_BASE_URL` when it starts, so the same image works with any backend without a rebuild.
* **Production details.** The backend runs as a non-root user, both images have Docker `HEALTHCHECK`s, the backend honors `PORT` for PaaS hosts, and `/health` returns 503 when the database is unreachable or no keys are loaded.
