import clsx from "clsx";
import { BarChart3, ChevronDown, ExternalLink, Flame, Quote } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { pct, truncate } from "../lib/format";
import type { DocumentResult } from "../lib/types";
import { VERDICT_META } from "../lib/verdict";
import { useCitations } from "./citationContext";
import { Chip, Meter, VerdictBadge } from "./ui";

export const isShown = (d: DocumentResult) => d.evidence_weight >= 0.5;

function MetaLine({ doc }: { doc: DocumentResult }) {
  const m = doc.meta ?? {};
  const parts = [m.journal, m.year].filter(Boolean) as string[];
  return (
    <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted">
      {parts.length > 0 && <span>{parts.join(" · ")}</span>}
      {m.authors && <span className="hidden truncate sm:inline">{truncate(m.authors, 60)}</span>}
      <Chip>{m.source ?? "SciFact"}</Chip>
      {m.pmid ? <Chip>PMID {m.pmid}</Chip> : doc.doc_id > 0 && !m.source ? <Chip>ID {doc.doc_id}</Chip> : null}
    </div>
  );
}

export function PaperCard({ doc, index }: { doc: DocumentResult; index: number }) {
  const { active } = useCitations();
  const [open, setOpen] = useState(false);
  const [heat, setHeat] = useState(false);
  const [details, setDetails] = useState(false);
  const ref = useRef<HTMLElement>(null);
  const shown = isShown(doc);
  const highlighted = active === doc.citation;

  useEffect(() => {
    const el = ref.current;
    const onFocus = () => setOpen(true);
    el?.addEventListener("cv:focus", onFocus);
    return () => el?.removeEventListener("cv:focus", onFocus);
  }, []);

  const evidenceIdx = new Set(doc.evidence.map((e) => e.index));
  const badge = shown ? (
    <VerdictBadge
      verdict={doc.stance}
      prob={doc.stance_probs[doc.stance]}
      label={VERDICT_META[doc.stance].stance + (doc.has_evidence ? "" : " · partial")}
      size="sm"
    />
  ) : (
    <span className="inline-block shrink-0 rounded-full bg-surface-2 px-2 py-0.5 text-[11.5px] font-medium text-muted">no evidence</span>
  );
  return (
    <motion.article
      ref={ref}
      id={`paper-${doc.citation}`}
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.3 }}
      className={clsx("card scroll-mt-24 p-4 transition-[box-shadow,border-color] sm:p-5", highlighted && "border-accent/60 ring-4 ring-accent/15")}
    >
      <header className="flex items-start gap-3">
        <span className="tnum grid h-7 min-w-7 shrink-0 place-items-center rounded-lg bg-surface-2 px-1.5 text-[13px] font-semibold text-ink-2">{doc.citation}</span>
        <div className="min-w-0 flex-1">
          <h3 className="text-[15px] leading-snug font-semibold tracking-tight text-ink">
            {doc.url ? (
              <a href={doc.url} target="_blank" rel="noreferrer" className="decoration-line-strong underline-offset-2 hover:underline">
                {doc.title}
              </a>
            ) : (
              doc.title
            )}
          </h3>
          <MetaLine doc={doc} />
          <div className="mt-2 sm:hidden">{badge}</div>
        </div>
        <div className="hidden sm:block">{badge}</div>
      </header>

      {shown ? (
        <div className="mt-3.5 space-y-2.5 pl-0 sm:pl-10">
          {doc.evidence.map((e) => {
            const m = VERDICT_META[e.stance];
            return (
              <figure key={e.index} className="rounded-xl border-l-[3px] px-3.5 py-2.5" style={{ borderColor: m.color, background: m.soft }}>
                <blockquote className="text-[14px] leading-relaxed text-ink">{e.text}</blockquote>
                <figcaption className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] text-muted">
                  <span className="inline-flex items-center gap-1.5">
                    Evidence <Meter value={e.rationale_score} color={m.color} className="w-12" />
                    <span className="tnum">{e.rationale_score.toFixed(2)}</span>
                  </span>
                  <span style={{ color: m.ink }} className="font-medium">
                    {m.stance} {pct(e.stance_probs[e.stance])}
                  </span>
                  <span>sentence {e.index + 1}</span>
                </figcaption>
              </figure>
            );
          })}
        </div>
      ) : (
        doc.evidence[0] && (
          <p className="mt-3 flex gap-2 pl-0 text-[13px] leading-relaxed text-muted sm:pl-10">
            <Quote className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>
              <span className="font-medium text-ink-2">Closest sentence</span> (relevance {doc.relevance.toFixed(2)}): {doc.evidence[0].text}
            </span>
          </p>
        )
      )}

      <div className="mt-3 flex flex-wrap items-center gap-1 pl-0 sm:pl-9">
        <button
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[12.5px] font-medium text-muted hover:bg-surface-2 hover:text-ink"
        >
          <ChevronDown className={clsx("h-3.5 w-3.5 transition-transform", open && "rotate-180")} />
          Full abstract <span className="tnum text-muted">({doc.sentences.length})</span>
        </button>
        <button
          onClick={() => setDetails((o) => !o)}
          aria-expanded={details}
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[12.5px] font-medium text-muted hover:bg-surface-2 hover:text-ink"
        >
          <BarChart3 className="h-3.5 w-3.5" /> Scores
        </button>
        {doc.url && (
          <a href={doc.url} target="_blank" rel="noreferrer" className="ml-auto inline-flex items-center gap-1 rounded-md px-2 py-1 text-[12.5px] font-medium text-accent-ink hover:bg-accent-soft">
            Open paper <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </div>

      <AnimatePresence initial={false}>
        {details && (
          <motion.dl
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1.5 overflow-hidden rounded-xl bg-surface-2 px-4 py-3 text-xs sm:ml-10 sm:grid-cols-3"
          >
            {[
              ["Rank after re-ranking", `#${doc.rank}`],
              ["Retriever rank", `#${doc.retrieval_rank || doc.rank}`],
              ["Cosine similarity", doc.dense_score.toFixed(3)],
              ["BM25 score", doc.bm25_score.toFixed(1)],
              ["Best rationale score", doc.relevance.toFixed(3)],
              ["Evidence weight", doc.evidence_weight.toFixed(2)],
              ["P(supports)", pct(doc.stance_probs.SUPPORTED, 1)],
              ["P(contradicts)", pct(doc.stance_probs.CONTRADICTED, 1)],
              ["P(neutral)", pct(doc.stance_probs.INSUFFICIENT_EVIDENCE, 1)],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-2">
                <dt className="text-muted">{k}</dt>
                <dd className="tnum font-medium text-ink">{v}</dd>
              </div>
            ))}
          </motion.dl>
        )}
      </AnimatePresence>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="mt-2 rounded-xl border border-line p-4 sm:ml-10">
              <div className="mb-2 flex items-center justify-between">
                <p className="text-xs font-medium tracking-wide text-muted uppercase">Abstract</p>
                <button
                  onClick={() => setHeat((h) => !h)}
                  aria-pressed={heat}
                  className={clsx(
                    "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11.5px] font-medium",
                    heat ? "bg-accent-soft text-accent-ink" : "text-muted hover:bg-surface-2",
                  )}
                >
                  <Flame className="h-3 w-3" /> Evidence heatmap
                </button>
              </div>
              <p className="text-[13.5px] leading-[1.75] text-ink-2">
                {doc.sentences.map((s, i) => {
                  const ev = doc.evidence.find((e) => e.index === i);
                  const score = doc.sentence_scores[i] ?? 0;
                  const style =
                    ev && shown
                      ? { background: VERDICT_META[ev.stance].soft, boxShadow: `inset 0 -2px 0 ${VERDICT_META[ev.stance].color}` }
                      : heat
                        ? { background: `color-mix(in oklab, var(--accent) ${Math.round(score * 38)}%, transparent)` }
                        : undefined;
                  return (
                    <span
                      key={i}
                      className={clsx("rounded-[3px] px-0.5 py-px box-decoration-clone", ev && shown && "text-ink", !evidenceIdx.has(i) && !heat && "")}
                      style={style}
                      title={`Sentence ${i + 1} · rationale score ${score.toFixed(2)}`}
                    >
                      {s}{" "}
                    </span>
                  );
                })}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.article>
  );
}
