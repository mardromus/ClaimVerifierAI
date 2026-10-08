import { Shuffle } from "lucide-react";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { truncate } from "../lib/format";
import type { ExampleClaim } from "../lib/types";
import { VERDICT_META } from "../lib/verdict";

const CURATED = [
  "LDL cholesterol has no involvement in the development of cardiovascular disease.",
  "32% of liver transplantation programs required patients to discontinue methadone treatment in 2001.",
  "Taking anti-depressants is associated with a decrease in the risk of gastrointestinal bleeding.",
];

export function ExampleChips({ onPick }: { onPick: (claim: string) => void }) {
  const [examples, setExamples] = useState<ExampleClaim[] | null>(null);
  const [seed, setSeed] = useState(7);
  useEffect(() => {
    api.examples(6, seed).then(setExamples).catch(() => setExamples([]));
  }, [seed]);
  const items = examples?.length ? examples : CURATED.map((claim, i) => ({ id: i, claim, label: undefined }));
  return (
    <div className="mt-5">
      <div className="mb-2.5 flex items-center justify-center gap-2 text-xs text-muted">
        <span>Try a claim from the SciFact dev set</span>
        <button onClick={() => setSeed((s) => s + 1)} className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 hover:bg-surface-2 hover:text-ink" aria-label="Show other examples">
          <Shuffle className="h-3 w-3" /> shuffle
        </button>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {items.map((ex, i) => (
          <motion.button
            key={`${seed}-${ex.id}`}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.04 }}
            onClick={() => onPick(ex.claim)}
            title={ex.label ? `Gold label in SciFact: ${VERDICT_META[ex.label].label}` : undefined}
            className="max-w-full rounded-full border border-line bg-surface px-3.5 py-1.5 text-left text-[13px] text-ink-2 shadow-soft transition-colors hover:border-line-strong hover:text-ink"
          >
            {truncate(ex.claim, 64)}
          </motion.button>
        ))}
      </div>
    </div>
  );
}
