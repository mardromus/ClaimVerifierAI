import { useState } from "react";
import { Tooltip } from "./ChartCard";
import { useWidth } from "./useWidth";

export interface BarSeries {
  name: string;
  color: string;
  values: number[]; // one per category, 0..1
}

/** Vertical grouped columns: <=24px wide, 4px rounded data end, 2px gap between neighbours. */
export function GroupedBars({ categories, series, height = 240 }: { categories: string[]; series: BarSeries[]; height?: number }) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ c: number; s: number } | null>(null);
  const m = { top: 18, right: 8, bottom: 28, left: 44 };
  const w = width - m.left - m.right;
  const h = height - m.top - m.bottom;
  const band = w / categories.length;
  const barW = Math.min(24, (band * 0.7 - (series.length - 1) * 2) / series.length);
  const groupW = series.length * barW + (series.length - 1) * 2;
  const y = (v: number) => m.top + h * (1 - v);
  const bx = (c: number, s: number) => m.left + c * band + (band - groupW) / 2 + s * (barW + 2);
  const bar = (x0: number, v: number) => {
    const top = y(v);
    const bottom = y(0);
    const r = Math.min(4, (bottom - top) / 2, barW / 2);
    return `M${x0},${bottom}V${top + r}Q${x0},${top} ${x0 + r},${top}H${x0 + barW - r}Q${x0 + barW},${top} ${x0 + barW},${top + r}V${bottom}Z`;
  };
  return (
    <div ref={ref} className="relative">
      <svg width={width} height={height} className="block" role="img" aria-label="Grouped bar chart">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={m.left} x2={m.left + w} y1={y(t)} y2={y(t)} stroke={t === 0 ? "var(--axis)" : "var(--grid)"} />
            <text x={m.left - 8} y={y(t)} dy="0.32em" textAnchor="end" className="tnum fill-[var(--muted)] text-[11px]">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        {categories.map((cat, c) => (
          <g key={cat}>
            <text x={m.left + c * band + band / 2} y={height - 8} textAnchor="middle" className="fill-[var(--text-2)] text-[11.5px]">
              {cat}
            </text>
            {series.map((s, si) => (
              <g key={s.name}>
                <path
                  d={bar(bx(c, si), s.values[c])}
                  fill={s.color}
                  opacity={hover && (hover.c !== c || hover.s !== si) ? 0.45 : 1}
                  className="transition-opacity"
                />
                {/* hit target larger than the mark */}
                <rect
                  x={bx(c, si) - 1}
                  y={m.top}
                  width={barW + 2}
                  height={h}
                  fill="transparent"
                  tabIndex={0}
                  aria-label={`${s.name}, ${cat}: ${(s.values[c] * 100).toFixed(1)}%`}
                  onPointerEnter={() => setHover({ c, s: si })}
                  onPointerLeave={() => setHover(null)}
                  onFocus={() => setHover({ c, s: si })}
                  onBlur={() => setHover(null)}
                />
              </g>
            ))}
          </g>
        ))}
      </svg>
      {hover && (
        <Tooltip x={bx(hover.c, hover.s) + barW / 2} y={y(series[hover.s].values[hover.c])} width={width}>
          <p className="tnum font-semibold text-ink">{(series[hover.s].values[hover.c] * 100).toFixed(1)}%</p>
          <p className="text-muted">
            {series[hover.s].name} · {categories[hover.c]}
          </p>
        </Tooltip>
      )}
    </div>
  );
}
