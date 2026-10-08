import { Info, Sparkles } from "lucide-react";
import { Fragment, useMemo } from "react";
import { splitCitations } from "../lib/citations";
import type { DocumentResult } from "../lib/types";
import { VERDICT_META } from "../lib/verdict";
import { useCitations } from "./citationContext";

function backendLabel(backend: string) {
  if (!backend) return "";
  if (backend.startsWith("template")) return backend.includes("unavailable") ? "Template (LLM unavailable)" : "Template";
  const [kind, model] = backend.split(":");
  const name = (model || kind).split("/").pop();
  return kind === "ollama" ? `${name} · Ollama` : name ?? backend;
}

export function CitationChip({ n, doc }: { n: number; doc?: DocumentResult }) {
  const { setActive, focus } = useCitations();
  const color = doc ? VERDICT_META[doc.evidence_weight >= 0.5 ? doc.stance : "INSUFFICIENT_EVIDENCE"].color : "var(--muted)";
  return (
    <button
      type="button"
      onMouseEnter={() => setActive(n)}
      onMouseLeave={() => setActive(null)}
      onFocus={() => setActive(n)}
      onBlur={() => setActive(null)}
      onClick={() => focus(n)}
      title={doc?.title}
      className="mx-0.5 inline-flex h-[18px] min-w-[18px] -translate-y-px items-center justify-center gap-1 rounded-md border border-line bg-surface-2 px-1 align-middle text-[11px] font-semibold text-ink-2 tabular-nums transition-colors hover:border-accent/50 hover:bg-accent-soft hover:text-accent-ink"
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
      {n}
    </button>
  );
}

export function ExplanationCard({
  text,
  backend,
  error,
  streaming,
  documents,
}: {
  text: string;
  backend: string;
  error?: string | null;
  streaming: boolean;
  documents: DocumentResult[];
}) {
  const byCitation = useMemo(() => new Map(documents.map((d) => [d.citation, d])), [documents]);
  const paragraphs = text.split(/\n{2,}/);
  return (
    <section className="card p-5 sm:p-6" aria-label="Explanation">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="flex items-center gap-2 text-[15px] font-semibold tracking-tight">
          <span className="grid h-6 w-6 place-items-center rounded-md bg-accent-soft text-accent-ink">
            <Sparkles className="h-3.5 w-3.5" />
          </span>
          Explanation
        </h2>
        {backend && (
          <span className="rounded-full border border-line bg-surface-2 px-2 py-0.5 text-[11px] font-medium text-muted" title={backend}>
            {backendLabel(backend)}
          </span>
        )}
      </div>
      {!text && streaming ? (
        <div className="space-y-2.5" aria-label="Generating explanation">
          <div className="shimmer h-3.5 w-11/12 rounded" />
          <div className="shimmer h-3.5 w-full rounded" />
          <div className="shimmer h-3.5 w-8/12 rounded" />
        </div>
      ) : (
        <div className="space-y-3 text-[15px] leading-[1.7] text-ink-2">
          {paragraphs.map((para, pi) => (
            <p key={pi}>
              {splitCitations(para).map((seg, i) =>
                seg.type === "text" ? (
                  <Fragment key={i}>{seg.text}</Fragment>
                ) : (
                  <CitationChip key={i} n={seg.n} doc={byCitation.get(seg.n)} />
                ),
              )}
              {streaming && pi === paragraphs.length - 1 && (
                <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-blink bg-accent" aria-hidden />
              )}
            </p>
          ))}
        </div>
      )}
      {error && !streaming && (
        <p className="mt-3 flex items-start gap-1.5 text-xs text-muted">
          <Info className="mt-px h-3.5 w-3.5 shrink-0" />
          The LLM could not be used ({error.length > 140 ? `${error.slice(0, 140)}…` : error}); showing an extractive explanation instead.
        </p>
      )}
    </section>
  );
}
