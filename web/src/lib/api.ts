import { SSEParser } from "./sse";
import type {
  ApiInfo,
  CandidatePreview,
  ExampleClaim,
  ExplanationEvent,
  ReportMetrics,
  ReportSummary,
  StageEvent,
  VerificationResult,
  VerifyRequest,
} from "./types";

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
  if (!res.ok) throw new ApiError(await errorMessage(res), res.status);
  return res.json() as Promise<T>;
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    /* not JSON */
  }
  return `${res.status} ${res.statusText}`;
}

export const api = {
  info: () => request<ApiInfo>("/api/info"),
  examples: (n = 12, seed?: number) =>
    request<ExampleClaim[]>(`/api/examples?n=${n}${seed !== undefined ? `&seed=${seed}` : ""}`),
  reports: () => request<ReportSummary[]>("/api/reports"),
  report: (name: string) => request<{ name: string; metrics: ReportMetrics }>(`/api/reports/${encodeURIComponent(name)}`),
  extractClaims: (text: string, maxClaims = 20) =>
    request<{ index: number; claim: string; score: number }[]>("/api/claims/extract", {
      method: "POST",
      body: JSON.stringify({ text, max_claims: maxClaims }),
    }),
  verifyBatch: (claims: string[], topK?: number) =>
    request<VerificationResult[]>("/api/verify/batch", {
      method: "POST",
      body: JSON.stringify({ claims, top_k: topK, explain: false }),
    }),
};

export interface StreamHandlers {
  onStage?: (e: StageEvent) => void;
  onCandidates?: (docs: CandidatePreview[]) => void;
  onResult?: (r: VerificationResult) => void;
  onToken?: (text: string) => void;
  onReset?: () => void;
  onExplanation?: (e: ExplanationEvent) => void;
  onDone?: (e: { timings_ms: Record<string, number>; total_ms: number }) => void;
}

/** POST /api/verify/stream and dispatch server-sent events as they arrive. */
export async function streamVerify(req: VerifyRequest, handlers: StreamHandlers, signal?: AbortSignal) {
  const res = await fetch("/api/verify/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok || !res.body) throw new ApiError(await errorMessage(res), res.status);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  const dispatch = (event: string, raw: string) => {
    const data = JSON.parse(raw);
    switch (event) {
      case "stage": return handlers.onStage?.(data);
      case "candidates": return handlers.onCandidates?.(data.documents);
      case "result": return handlers.onResult?.(data);
      case "token": return handlers.onToken?.(data.text);
      case "reset": return handlers.onReset?.();
      case "explanation": return handlers.onExplanation?.(data);
      case "done": return handlers.onDone?.(data);
      case "error": throw new ApiError(data.message);
    }
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const msg of parser.push(decoder.decode(value, { stream: true }))) dispatch(msg.event, msg.data);
  }
  for (const msg of parser.flush()) dispatch(msg.event, msg.data);
}
