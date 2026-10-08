import { AlertTriangle } from "lucide-react";
import { motion } from "motion/react";
import { pct } from "../lib/format";
import type { VerificationResult, Verdict } from "../lib/types";
import { VERDICTS, VERDICT_META } from "../lib/verdict";

const SOURCE_LABEL: Record<string, string> = {
  corpus: "the SciFact corpus",
  europepmc: "Europe PMC",
  pubmed: "PubMed",
  custom: "your abstracts",
};

function ConfidenceRing({ value, verdict }: { value: number; verdict: Verdict }) {
  const r = 42;
  const c = 2 * Math.PI * r;
  const m = VERDICT_META[verdict];
  return (
    <div className="relative h-28 w-28 shrink-0" role="img" aria-label={`Confidence ${pct(value)}`}>
      <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
        <circle cx="50" cy="50" r={r} fill="none" stroke="var(--surface-3)" strokeWidth="8" />
        <motion.circle
          cx="50"
          cy="50"
          r={r}
          fill="none"
          stroke={m.color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - value) }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">
        <div>
          <p className="text-2xl font-semibold tracking-tight">{Math.round(value * 100)}<span className="text-sm text-muted">%</span></p>
          <p className="text-[10.5px] font-medium tracking-wide text-muted uppercase">confidence</p>
        </div>
      </div>
    </div>
  );
}

/** Three-way probability split: stacked bar with 2px surface gaps, legend always present. */
export function ProbabilityBar({ scores }: { scores: Record<Verdict, number> }) {
  return (
    <div>
      <div className="flex h-2.5 w-full gap-[2px] overflow-hidden rounded-full" role="img" aria-label="Probability of each verdict">
        {VERDICTS.map((v) => (
          <motion.span
            key={v}
            className="h-full first:rounded-l-full last:rounded-r-full"
            style={{ background: VERDICT_META[v].color }}
            initial={{ width: 0 }}
            animate={{ width: `${scores[v] * 100}%` }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            title={`${VERDICT_META[v].label}: ${pct(scores[v], 1)}`}
          />
        ))}
      </div>
      <ul className="mt-2.5 grid grid-cols-1 gap-1.5 text-xs sm:grid-cols-3 sm:gap-2">
        {VERDICTS.map((v) => {
          const Icon = VERDICT_META[v].icon;
          return (
            <li key={v} className="flex min-w-0 items-center gap-1.5">
              <Icon className="h-3.5 w-3.5 shrink-0" style={{ color: VERDICT_META[v].color }} />
              <span className="truncate text-muted">{VERDICT_META[v].short}</span>
              <span className="tnum ml-auto font-medium text-ink">{pct(scores[v], 1)}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Strength({ label, value, verdict }: { label: string; value: number; verdict: Verdict }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-muted">{label}</span>
        <span className="tnum font-medium">{value.toFixed(2)}</span>
      </div>
      <div className="h-1.5 rounded-full bg-surface-3">
        <motion.div
          className="h-full rounded-full"
          style={{ background: VERDICT_META[verdict].color }}
          initial={{ width: 0 }}
          animate={{ width: `${Math.max(1, value * 100)}%` }}
          transition={{ duration: 0.7, delay: 0.15 }}
        />
      </div>
    </div>
  );
}

export function VerdictCard({ result }: { result: VerificationResult }) {
  const m = VERDICT_META[result.verdict];
  const Icon = m.icon;
  const withEvidence = result.documents.filter((d) => d.evidence_weight >= 0.5).length;
  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="card relative overflow-hidden"
      aria-label="Verdict"
    >
      <div className="absolute inset-x-0 top-0 h-1" style={{ background: m.color }} />
      <div className="pointer-events-none absolute -top-24 -right-24 h-64 w-64 rounded-full opacity-60 blur-3xl" style={{ background: m.soft }} />
      <div className="relative flex flex-col gap-6 p-5 sm:flex-row sm:items-center sm:p-6">
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium tracking-wide text-muted uppercase">Verdict</p>
          <div className="mt-1.5 flex items-center gap-2.5">
            <span className="grid h-10 w-10 place-items-center rounded-full" style={{ background: m.soft, color: m.ink }}>
              <Icon className="h-6 w-6" strokeWidth={2.2} />
            </span>
            <h2 className="text-[28px] leading-tight font-semibold tracking-tight sm:text-3xl" style={{ color: m.ink }}>
              {m.label}
            </h2>
          </div>
          <p className="mt-2 text-[13.5px] leading-relaxed text-ink-2">
            {m.description} Based on <b className="font-semibold">{result.documents.length} papers</b> from{" "}
            {SOURCE_LABEL[result.source] ?? result.source}
            {result.candidates_scanned > result.documents.length && <> (re-ranked from {result.candidates_scanned} candidates)</>};{" "}
            {withEvidence} contain{withEvidence === 1 ? "s" : ""} relevant evidence.
          </p>
          {result.mixed_evidence && (
            <p className="mt-2 inline-flex items-center gap-1.5 rounded-lg bg-surface-2 px-2.5 py-1 text-xs font-medium text-ink-2">
              <AlertTriangle className="h-3.5 w-3.5" /> Mixed evidence: some papers support and others contradict the claim.
            </p>
          )}
        </div>
        <ConfidenceRing value={result.confidence} verdict={result.verdict} />
      </div>
      <div className="relative grid gap-5 border-t border-line px-5 py-4 sm:grid-cols-[1.4fr_1fr] sm:px-6">
        <ProbabilityBar scores={result.scores} />
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-1 sm:gap-2.5">
          <Strength label="Strongest support" value={result.support_strength} verdict="SUPPORTED" />
          <Strength label="Strongest contradiction" value={result.contradict_strength} verdict="CONTRADICTED" />
        </div>
      </div>
    </motion.section>
  );
}
