import clsx from "clsx";
import { Check, Loader2, Scale, ScanText, Search, Sparkles } from "lucide-react";
import { motion } from "motion/react";
import type { StageState } from "../hooks/useVerifyStream";
import { ms } from "../lib/format";
import type { StageName } from "../lib/types";

const STEPS: { key: StageName; label: string; icon: typeof Search; model: string }[] = [
  { key: "retrieval", label: "Retrieve", icon: Search, model: "Semantic + BM25 search" },
  { key: "rationale", label: "Select evidence", icon: ScanText, model: "Rationale selector" },
  { key: "nli", label: "Infer stance", icon: Scale, model: "NLI verifier" },
  { key: "explanation", label: "Explain", icon: Sparkles, model: "LLM, RAG prompt" },
];

export function PipelineProgress({ stages, explain, totalMs }: { stages: Record<StageName, StageState>; explain: boolean; totalMs: number | null }) {
  const steps = explain ? STEPS : STEPS.slice(0, 3);
  const progress = steps.reduce((acc, s) => acc + (stages[s.key].status === "done" ? 1 : stages[s.key].status === "running" ? 0.5 : 0), 0) / steps.length;
  return (
    <div className="card relative overflow-hidden px-4 py-3.5 sm:px-5">
      <motion.div
        className="absolute inset-x-0 top-0 h-0.5 origin-left bg-accent"
        initial={false}
        animate={{ scaleX: progress, opacity: progress >= 1 ? 0 : 1 }}
        transition={{ duration: 0.45, ease: "easeOut" }}
      />
      <ol className="grid grid-cols-2 gap-3 sm:grid-cols-[repeat(var(--n),minmax(0,1fr))]" style={{ ["--n" as string]: steps.length }}>
        {steps.map((step) => {
          const s = stages[step.key];
          const Icon = step.icon;
          return (
            <li key={step.key} className="min-w-0">
              <div className="flex items-center gap-2.5">
                <span
                  className={clsx(
                    "relative grid h-8 w-8 shrink-0 place-items-center rounded-full border transition-colors",
                    s.status === "done" && "border-transparent bg-accent text-white",
                    s.status === "running" && "animate-pulse-ring border-accent/50 bg-accent-soft text-accent-ink",
                    s.status === "pending" && "border-line bg-surface-2 text-muted",
                  )}
                >
                  {s.status === "done" ? <Check className="h-4 w-4" strokeWidth={2.5} /> : s.status === "running" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Icon className="h-4 w-4" />}
                </span>
                <div className="min-w-0">
                  <p className={clsx("truncate text-[13px] font-medium", s.status === "pending" ? "text-muted" : "text-ink")}>{step.label}</p>
                  <p className="truncate text-[11.5px] text-muted">
                    {s.status === "done" && s.ms !== undefined ? <span className="tnum">{ms(s.ms)}</span> : null}
                    {s.status === "done" && s.ms !== undefined && s.detail ? " · " : null}
                    {s.detail ?? step.model}
                  </p>
                </div>
              </div>
            </li>
          );
        })}
      </ol>
      {totalMs !== null && (
        <p className="mt-2.5 border-t border-line pt-2 text-right text-[11.5px] text-muted">
          Completed in <span className="tnum font-medium text-ink-2">{ms(totalMs)}</span>
        </p>
      )}
    </div>
  );
}
