export type ResolutionSource = "LIVE_INFERENCE" | "LLM_CACHE";

export interface ChatMetadata {
  resolution_source: ResolutionSource;
  model: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  key_id: string | null;
  fallback_attempts: number;
}

export interface ChatResponse {
  session_id: string;
  reply: string;
  metadata: ChatMetadata;
}

export interface Health {
  status: string;
  keys_configured: number;
  cache_enabled: boolean;
  failure_simulation?: boolean;
  models?: string[];
}

declare global {
  interface Window {
    __APP_CONFIG__?: { API_BASE_URL?: string };
  }
}

export const API_BASE_URL: string = (
  window.__APP_CONFIG__?.API_BASE_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  "http://localhost:8000"
).replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export function newSessionId(): string {
  const rand =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID().replace(/-/g, "")
      : Math.random().toString(16).slice(2) + Date.now().toString(16);
  return `s-${rand.slice(0, 12)}`;
}

/** Turn any FastAPI error body into one readable sentence. */
function describeError(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (status === 503) {
    return "All AI providers are unavailable right now (every API key failed). Please try again in a moment.";
  }
  if (status === 422) {
    if (Array.isArray(detail)) {
      const parts = detail.map((d: { loc?: unknown[]; msg?: string }) => {
        const field = (d.loc || []).filter((p) => p !== "body").join(".");
        const msg = (d.msg || "invalid value").replace(/^Value error, /, "");
        return field ? `${field}: ${msg}` : msg;
      });
      return `Invalid request — ${parts.join("; ")}`;
    }
    return `Invalid request — ${typeof detail === "string" ? detail : "please check your message"}`;
  }
  if (typeof detail === "string") return `Error ${status}: ${detail}`;
  return `Unexpected error (HTTP ${status}).`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(0, `Could not reach the backend at ${API_BASE_URL}. Check that it is running.`);
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) throw new ApiError(res.status, describeError(res.status, body));
  return body as T;
}

export function sendChat(
  sessionId: string,
  message: string,
  systemPrompt?: string,
  simulateFailure?: string,
): Promise<ChatResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (simulateFailure) headers["X-Simulate-Failure"] = simulateFailure;
  const payload: Record<string, string> = { session_id: sessionId, message };
  if (systemPrompt && systemPrompt.trim()) payload.system_prompt = systemPrompt.trim();
  return request<ChatResponse>("/chat", { method: "POST", headers, body: JSON.stringify(payload) });
}

export function getHealth(): Promise<Health> {
  return request<Health>("/health");
}
