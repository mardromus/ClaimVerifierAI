import { FileSearch } from "lucide-react";
import { motion } from "motion/react";
import { truncate } from "../lib/format";
import type { CandidatePreview } from "../lib/types";

export function CandidateList({ candidates, running }: { candidates: CandidatePreview[]; running: boolean }) {
  if (!candidates.length) {
    return (
      <div className="card space-y-3 p-5" aria-busy="true" aria-label="Searching the literature">
        {[0, 1, 2].map((i) => (
          <div key={i} className="flex items-center gap-3">
            <div className="shimmer h-7 w-7 rounded-lg" />
            <div className="flex-1 space-y-1.5">
              <div className="shimmer h-3.5 w-3/4 rounded" />
              <div className="shimmer h-3 w-1/3 rounded" />
            </div>
          </div>
        ))}
      </div>
    );
  }
  return (
    <section className="card p-5" aria-label="Candidate papers">
      <h2 className="mb-3 flex items-center gap-2 text-[13.5px] font-semibold">
        <FileSearch className="h-4 w-4 text-muted" />
        {running ? "Reading" : "Scanned"} {candidates.length} candidate papers
      </h2>
      <ul className="space-y-1.5">
        {candidates.slice(0, 12).map((c, i) => (
          <motion.li
            key={`${c.doc_id}-${i}`}
            initial={{ opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.03 }}
            className="flex items-center gap-3 rounded-lg px-1 py-1"
          >
            <span className="tnum grid h-6 w-6 shrink-0 place-items-center rounded-md bg-surface-2 text-[11px] font-semibold text-muted">{c.rank}</span>
            <span className="min-w-0 flex-1 truncate text-[13px] text-ink-2">{truncate(c.title, 110)}</span>
            <span className="hidden shrink-0 text-[11px] text-muted sm:inline">
              {[c.meta?.journal && truncate(c.meta.journal, 24), c.meta?.year].filter(Boolean).join(" · ") || `${c.num_sentences} sentences`}
            </span>
          </motion.li>
        ))}
      </ul>
    </section>
  );
}
