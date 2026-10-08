import { useState } from "react";

/** Confusion-matrix heatmap: one-hue sequential ramp by row share (recall), counts printed in cells. */
export function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const [hover, setHover] = useState<[number, number] | null>(null);
  const rowTotals = matrix.map((r) => r.reduce((a, b) => a + b, 0) || 1);
  return (
    <div className="overflow-x-auto">
      <div className="grid min-w-[360px] gap-[2px]" style={{ gridTemplateColumns: `120px repeat(${labels.length}, minmax(0, 1fr))` }}>
        <div className="flex items-end pb-1 text-[11px] text-muted">gold ↓ · predicted →</div>
        {labels.map((l) => (
          <div key={l} className="px-1 pb-1 text-center text-[11.5px] font-medium text-ink-2">{l}</div>
        ))}
        {matrix.map((row, i) => (
          <div key={i} className="contents">
            <div className="flex items-center pr-2 text-[11.5px] font-medium text-ink-2">{labels[i]}</div>
            {row.map((v, j) => {
              const share = v / rowTotals[i];
              const dark = share > 0.72;
              return (
                <div
                  key={j}
                  tabIndex={0}
                  onPointerEnter={() => setHover([i, j])}
                  onPointerLeave={() => setHover(null)}
                  onFocus={() => setHover([i, j])}
                  onBlur={() => setHover(null)}
                  aria-label={`gold ${labels[i]}, predicted ${labels[j]}: ${v} (${Math.round(share * 100)}% of row)`}
                  className="relative grid h-16 place-items-center rounded-md transition-[outline] outline-offset-[-2px] hover:outline-2 hover:outline-[var(--text)]"
                  style={{ background: `color-mix(in oklab, var(--seq-1) ${Math.round(8 + share * 92)}%, var(--seq-0))` }}
                >
                  <span className={`tnum text-[15px] font-semibold ${dark ? "text-white" : "text-ink"}`}>{v}</span>
                  <span className={`tnum text-[10.5px] ${dark ? "text-white/80" : "text-muted"}`}>{Math.round(share * 100)}%</span>
                  {hover && hover[0] === i && hover[1] === j && i === j && <span className="sr-only">correct</span>}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <p className="mt-2 text-[11.5px] text-muted">Cell shade = share of the gold row (recall); the diagonal holds correct predictions.</p>
    </div>
  );
}
