import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { API_BASE_URL, ApiError, ChatMetadata, Health, getHealth, newSessionId, sendChat } from "./api";

type Entry =
  | { id: number; kind: "user"; content: string; simulated?: string }
  | { id: number; kind: "assistant"; content: string; meta: ChatMetadata }
  | { id: number; kind: "error"; content: string; retryMessage: string };

const MAX_CHARS = 4000;

const SIMULATION_PRESETS: { label: string; value: string }[] = [
  { label: "Off", value: "" },
  { label: "key_1 → 429 (rate limit)", value: "key_1=429" },
  { label: "key_1 → 500 (server error)", value: "key_1=500" },
  { label: "key_1 → 429, key_2 → 500", value: "key_1=429,key_2=500" },
  { label: "All keys → 429", value: "*=429" },
];

let nextId = 1;

export default function App() {
  // A new session starts on every page load.
  const [sessionId, setSessionId] = useState(newSessionId);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulate, setSimulate] = useState("");
  const [pending, setPending] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    getHealth()
      .then((h) => {
        setHealth(h);
        setHealthError(null);
      })
      .catch((e: unknown) => setHealthError(e instanceof Error ? e.message : "Backend unreachable"));
  }, []);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [entries, pending]);

  async function send(text: string) {
    const message = text.trim();
    if (!message || pending) return;
    const sim = simulate.trim();
    setEntries((prev) => [...prev, { id: nextId++, kind: "user", content: message, simulated: sim || undefined }]);
    setPending(true);
    try {
      const res = await sendChat(sessionId, message, systemPrompt, sim || undefined);
      setEntries((prev) => [...prev, { id: nextId++, kind: "assistant", content: res.reply, meta: res.metadata }]);
    } catch (e: unknown) {
      const content = e instanceof ApiError || e instanceof Error ? e.message : "Something went wrong.";
      setEntries((prev) => [...prev, { id: nextId++, kind: "error", content, retryMessage: message }]);
    } finally {
      setPending(false);
      inputRef.current?.focus();
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input;
    if (!text.trim() || pending) return;
    setInput("");
    void send(text);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      onSubmit(e);
    }
  }

  function newChat() {
    setSessionId(newSessionId());
    setEntries([]);
    setInput("");
    inputRef.current?.focus();
  }

  const totalChars = input.trim().length;
  const overLimit = totalChars > MAX_CHARS;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>
            ◆
          </span>
          <div>
            <h1>Lyftr Chat</h1>
            <p className="subtitle">
              Session <code data-testid="session-id">{sessionId}</code>
            </p>
          </div>
        </div>
        <div className="topbar-actions">
          <HealthBadge health={health} error={healthError} />
          <button type="button" className="ghost" onClick={() => setShowSettings((s) => !s)} aria-expanded={showSettings}>
            {showSettings ? "Hide settings" : "Settings"}
          </button>
          <button type="button" className="ghost" onClick={newChat} data-testid="new-chat-button">
            New chat
          </button>
        </div>
      </header>

      {showSettings && (
        <section className="settings" aria-label="Session settings">
          <label className="field">
            <span>System prompt (this session)</span>
            <textarea
              data-testid="system-prompt-input"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Leave empty to use the default — e.g. “Answer like a pirate.”"
              rows={2}
            />
          </label>
          <label className="field">
            <span>
              Simulate failure <small>(sends X-Simulate-Failure on the next messages)</small>
            </span>
            <div className="sim-row">
              <select
                data-testid="simulate-failure-select"
                value={SIMULATION_PRESETS.some((p) => p.value === simulate) ? simulate : "__custom"}
                onChange={(e) => setSimulate(e.target.value === "__custom" ? simulate || "key_2=503" : e.target.value)}
              >
                {SIMULATION_PRESETS.map((p) => (
                  <option key={p.value} value={p.value}>
                    {p.label}
                  </option>
                ))}
                <option value="__custom">Custom…</option>
              </select>
              <input
                type="text"
                value={simulate}
                onChange={(e) => setSimulate(e.target.value)}
                placeholder="key_1=429,key_2=500"
                aria-label="X-Simulate-Failure header value"
              />
            </div>
            {health && health.failure_simulation === false && simulate && (
              <small className="warn">The backend has ENABLE_FAILURE_SIMULATION off, so this header is ignored.</small>
            )}
          </label>
        </section>
      )}

      <main className="messages" ref={listRef} aria-live="polite">
        {entries.length === 0 && !pending && (
          <div className="empty">
            <h2>Start a conversation</h2>
            <p>
              Each reply shows where it came from (<b>LIVE_INFERENCE</b> or <b>LLM_CACHE</b>), the model, latency and
              that request's token usage. Ask the same first question in a new chat to see a cache hit.
            </p>
            <div className="suggestions">
              {["What is the capital of France?", "Explain caching in one sentence.", "Give me a haiku about failover."].map(
                (s) => (
                  <button key={s} type="button" className="chip-btn" onClick={() => void send(s)}>
                    {s}
                  </button>
                ),
              )}
            </div>
          </div>
        )}

        {entries.map((entry) => {
          if (entry.kind === "user") {
            return (
              <div key={entry.id} className="row user" data-testid="user-message">
                <div className="bubble">
                  {entry.content}
                  {entry.simulated && <div className="sim-note">X-Simulate-Failure: {entry.simulated}</div>}
                </div>
              </div>
            );
          }
          if (entry.kind === "error") {
            return (
              <div key={entry.id} className="row assistant" role="alert" data-testid="error-message">
                <div className="bubble error">
                  <strong>Request failed.</strong> {entry.content}
                  <div>
                    <button type="button" className="link" onClick={() => void send(entry.retryMessage)} disabled={pending}>
                      Retry
                    </button>
                  </div>
                </div>
              </div>
            );
          }
          return <AssistantMessage key={entry.id} content={entry.content} meta={entry.meta} />;
        })}

        {pending && (
          <div className="row assistant" aria-label="Assistant is typing">
            <div className="bubble typing">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}
      </main>

      <form className="composer" onSubmit={onSubmit}>
        <textarea
          ref={inputRef}
          data-testid="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type a message… (Enter to send, Shift+Enter for a new line)"
          rows={1}
          autoFocus
        />
        <div className="composer-side">
          <span className={`counter ${overLimit ? "over" : ""}`}>
            {totalChars}/{MAX_CHARS}
          </span>
          <button type="submit" data-testid="send-button" disabled={pending || !input.trim()}>
            {pending ? "Sending…" : "Send"}
          </button>
        </div>
      </form>
      <footer className="footnote">
        Backend: <code>{API_BASE_URL}</code>
      </footer>
    </div>
  );
}

function AssistantMessage({ content, meta }: { content: string; meta: ChatMetadata }) {
  const cached = meta.resolution_source === "LLM_CACHE";
  return (
    <div className="row assistant" data-testid="assistant-message">
      <div className="bubble">
        <div className="reply">{content}</div>
        <div className="meta">
          <span className={`tag ${cached ? "cache" : "live"}`} title="Resolution source">
            <span data-testid="meta-source">{meta.resolution_source}</span>
          </span>
          <span className="tag" title="Model">
            <span data-testid="meta-model">{meta.model}</span>
          </span>
          <span className="tag" title="Latency">
            <span data-testid="meta-latency">{meta.latency_ms} ms</span>
          </span>
          <span className="tag" title="Input tokens for this request">
            in <span data-testid="meta-input-tokens">{meta.input_tokens}</span>
          </span>
          <span className="tag" title="Output tokens for this reply">
            out <span data-testid="meta-output-tokens">{meta.output_tokens}</span>
          </span>
          <span className="tag subtle" title="API key that answered">
            key <span data-testid="meta-key-id">{meta.key_id ?? "—"}</span>
          </span>
          {meta.fallback_attempts > 0 && (
            <span className="tag warn" title="Keys that failed before this one answered">
              failover × <span data-testid="meta-fallback-attempts">{meta.fallback_attempts}</span>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

function HealthBadge({ health, error }: { health: Health | null; error: string | null }) {
  if (error) {
    return (
      <span className="health down" title={error}>
        ● backend offline
      </span>
    );
  }
  if (!health) return <span className="health">● checking…</span>;
  return (
    <span className="health up" title={(health.models || []).join(", ")}>
      ● {health.keys_configured} keys · cache {health.cache_enabled ? "on" : "off"}
    </span>
  );
}
