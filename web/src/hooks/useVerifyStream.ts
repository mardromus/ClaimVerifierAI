import { useCallback, useRef, useState } from "react";
import { streamVerify } from "../lib/api";
import type {
  CandidatePreview,
  ExplanationEvent,
  StageName,
  VerificationResult,
  VerifyRequest,
} from "../lib/types";

export type StageStatus = "pending" | "running" | "done";
export interface StageState {
  status: StageStatus;
  ms?: number;
  detail?: string;
}

export interface VerifyState {
  status: "idle" | "running" | "done" | "error";
  request: VerifyRequest | null;
  stages: Record<StageName, StageState>;
  candidates: CandidatePreview[];
  result: VerificationResult | null;
  explanation: string;
  explanationFinal: ExplanationEvent | null;
  error: string | null;
  totalMs: number | null;
  startedAt: number | null;
}

const freshStages = (): Record<StageName, StageState> => ({
  retrieval: { status: "pending" },
  rationale: { status: "pending" },
  nli: { status: "pending" },
  explanation: { status: "pending" },
});

const initial: VerifyState = {
  status: "idle",
  request: null,
  stages: freshStages(),
  candidates: [],
  result: null,
  explanation: "",
  explanationFinal: null,
  error: null,
  totalMs: null,
  startedAt: null,
};

function stageDetail(e: { stage: StageName; status: string; count?: number; source?: string; sentences?: number; kept?: number; backend?: string }) {
  if (e.stage === "retrieval") return e.status === "start" ? `Searching ${e.source ?? ""}` : `${e.count} papers found`;
  if (e.stage === "rationale")
    return e.status === "start" ? `Scoring sentences in ${e.count} papers` : `${e.sentences} sentences · kept ${e.kept}`;
  if (e.stage === "nli") return e.status === "start" ? "Checking entailment" : "Verdict ready";
  if (e.stage === "explanation") return e.status === "start" ? `Writing with ${e.backend}` : "Explanation ready";
  return "";
}

export function useVerifyStream(onComplete?: (r: VerificationResult) => void) {
  const [state, setState] = useState<VerifyState>(initial);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(
    async (request: VerifyRequest) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setState({ ...initial, stages: freshStages(), status: "running", request, startedAt: Date.now() });
      let final: VerificationResult | null = null;
      try {
        await streamVerify(
          request,
          {
            onStage: (e) =>
              setState((s) => ({
                ...s,
                stages: { ...s.stages, [e.stage]: { status: e.status === "start" ? "running" : "done", ms: e.ms, detail: stageDetail(e) } },
              })),
            onCandidates: (candidates) => setState((s) => ({ ...s, candidates })),
            onResult: (result) => {
              final = result;
              setState((s) => ({ ...s, result }));
            },
            onToken: (text) => setState((s) => ({ ...s, explanation: s.explanation + text })),
            onReset: () => setState((s) => ({ ...s, explanation: "" })),
            onExplanation: (e) => {
              if (final) final = { ...final, explanation: e.text, explanation_backend: e.backend, citations: e.citations };
              setState((s) => ({
                ...s,
                explanation: e.text,
                explanationFinal: e,
                result: s.result ? { ...s.result, explanation: e.text, explanation_backend: e.backend, citations: e.citations } : s.result,
              }));
            },
            onDone: (e) => {
              if (final) final = { ...final, timings_ms: e.timings_ms };
              setState((s) => ({
                ...s,
                status: "done",
                totalMs: e.total_ms,
                result: s.result ? { ...s.result, timings_ms: e.timings_ms } : s.result,
              }));
            },
          },
          controller.signal,
        );
        if (final) onComplete?.(final);
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setState((s) => ({ ...s, status: "error", error: (err as Error).message || "Verification failed" }));
      }
    },
    [onComplete],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setState((s) => (s.status === "running" ? { ...s, status: s.result ? "done" : "idle" } : s));
  }, []);

  const show = useCallback((result: VerificationResult) => {
    abortRef.current?.abort();
    const done: StageState = { status: "done" };
    setState({
      ...initial,
      status: "done",
      request: { claim: result.claim, source: result.source },
      stages: { retrieval: done, rationale: done, nli: done, explanation: result.explanation ? done : { status: "pending" } },
      result,
      explanation: result.explanation,
      explanationFinal: result.explanation
        ? { text: result.explanation, backend: result.explanation_backend, citations: result.citations }
        : null,
    });
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(initial);
  }, []);

  return { state, run, cancel, show, reset };
}
