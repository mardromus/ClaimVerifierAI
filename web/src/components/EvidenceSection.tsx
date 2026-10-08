import clsx from "clsx";
import { Library } from "lucide-react";
import { useMemo, useState } from "react";
import type { DocumentResult, Verdict } from "../lib/types";
import { VERDICT_META } from "../lib/verdict";
import { isShown, PaperCard } from "./PaperCard";

type Filter = "all" | Verdict | "none";

export function EvidenceSection({ documents }: { documents: DocumentResult[] }) {
  const [filter, setFilter] = useState<Filter>("all");
  const strength = (d: DocumentResult) => d.evidence_weight * Math.max(d.stance_probs.SUPPORTED, d.stance_probs.CONTRADICTED);
  const ordered = useMemo(() => {
    const shown = documents.filter(isShown).sort((a, b) => strength(b) - strength(a));
    return [...shown, ...documents.filter((d) => !isShown(d))];
  }, [documents]);
  const counts = {
    SUPPORTED: documents.filter((d) => isShown(d) && d.stance === "SUPPORTED").length,
    CONTRADICTED: documents.filter((d) => isShown(d) && d.stance === "CONTRADICTED").length,
    INSUFFICIENT_EVIDENCE: documents.filter((d) => isShown(d) && d.stance === "INSUFFICIENT_EVIDENCE").length,
    none: documents.filter((d) => !isShown(d)).length,
  };
  const visible = ordered.filter((d) =>
    filter === "all" ? true : filter === "none" ? !isShown(d) : isShown(d) && d.stance === filter,
  );
  const tabs: { key: Filter; label: string; count: number; color?: string }[] = [
    { key: "all", label: "All", count: documents.length },
    { key: "SUPPORTED", label: "Supports", count: counts.SUPPORTED, color: VERDICT_META.SUPPORTED.color },
    { key: "CONTRADICTED", label: "Contradicts", count: counts.CONTRADICTED, color: VERDICT_META.CONTRADICTED.color },
    { key: "INSUFFICIENT_EVIDENCE", label: "Neutral", count: counts.INSUFFICIENT_EVIDENCE, color: VERDICT_META.INSUFFICIENT_EVIDENCE.color },
    { key: "none", label: "No evidence", count: counts.none },
  ];
  return (
    <section aria-label="Evidence">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 text-[15px] font-semibold tracking-tight">
          <Library className="h-4 w-4 text-muted" /> Evidence from {documents.length} papers
        </h2>
        <div className="scrollbar-thin flex max-w-full gap-1 overflow-x-auto rounded-xl border border-line bg-surface p-1" role="tablist">
          {tabs
            .filter((t) => t.key === "all" || t.count > 0)
            .map((t) => (
              <button
                key={t.key}
                role="tab"
                aria-selected={filter === t.key}
                onClick={() => setFilter(t.key)}
                className={clsx(
                  "inline-flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1 text-[12.5px] font-medium transition-colors",
                  filter === t.key ? "bg-surface-2 text-ink" : "text-muted hover:text-ink",
                )}
              >
                {t.color && <span className="h-2 w-2 rounded-full" style={{ background: t.color }} />}
                {t.label}
                <span className="tnum text-muted">{t.count}</span>
              </button>
            ))}
        </div>
      </div>
      <div className="space-y-3">
        {visible.map((d, i) => (
          <PaperCard key={`${d.doc_id}-${d.citation}`} doc={d} index={i} />
        ))}
      </div>
    </section>
  );
}
