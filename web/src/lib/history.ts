import type { SourceName, VerificationResult, Verdict } from "./types";

export interface HistoryEntry {
  id: string;
  claim: string;
  verdict: Verdict;
  confidence: number;
  source: SourceName;
  ts: number;
  result?: VerificationResult;
}

const KEY = "claimverifier.history.v1";
const MAX = 30;

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

export function saveHistory(entries: HistoryEntry[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(entries.slice(0, MAX)));
  } catch {
    // Quota exceeded or storage unavailable: drop stored results, keep the summaries.
    try {
      localStorage.setItem(KEY, JSON.stringify(entries.slice(0, MAX).map(({ result: _r, ...e }) => e)));
    } catch {
      /* storage unavailable (private mode) - history is a convenience only */
    }
  }
}

export function addToHistory(result: VerificationResult): HistoryEntry[] {
  const entry: HistoryEntry = {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
    claim: result.claim,
    verdict: result.verdict,
    confidence: result.confidence,
    source: result.source,
    ts: Date.now(),
    result,
  };
  const entries = [entry, ...loadHistory().filter((e) => e.claim !== result.claim || e.source !== result.source)];
  saveHistory(entries);
  return entries.slice(0, MAX);
}

export function clearHistory() {
  saveHistory([]);
}
