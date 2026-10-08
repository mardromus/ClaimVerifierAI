import { useState } from "react";
import { Tooltip, TooltipRow } from "./ChartCard";
import { useWidth } from "./useWidth";

export interface LineSeries {
  name: string;
  color: string;
  values: number[]; // 0..1
}

/** Lines over ordinal x positions with a crosshair tooltip listing every series. */
export function LineChart({ xLabels, series, height = 220 }: { xLabels: string[]; series: LineSeries[]; height?: number }) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const m = { top: 12, right: 16, bottom: 26, left: 44 };
  const h = height - m.top - m.bottom;
  const y = (v: number) => m.top + h * (1 - v);
  const endYs = series.map((s) => y(s.values[s.values.length - 1])).sort((a, b) => a - b);
  const labelEnds = endYs.every((v, i) => i === 0 || v - endYs[i - 1] >= 16);
  const w = width - m.left - (labelEnds ? 92 : m.right);
  const x = (i: number) => m.left + (xLabels.length === 1 ? w / 2 : (i * w) / (xLabels.length - 1));
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const nearest = (px: number) => {
    let best = 0;
    xLabels.forEach((_, i) => Math.abs(x(i) - px) < Math.abs(x(best) - px) && (best = i));
    return best;
  };
  return (
    <div ref={ref} className="relative">
      <svg
        width={width}
        height={height}
        className="block touch-none"
        role="img"
        aria-label={`Line chart: ${series.map((s) => s.name).join(", ")}`}
        onPointerMove={(e) => setHover(nearest(e.clientX - e.currentTarget.getBoundingClientRect().left))}
        onPointerLeave={() => setHover(null)}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.left} x2={m.left + w} y1={y(t)} y2={y(t)} stroke={t === 0 ? "var(--axis)" : "var(--grid)"} strokeWidth={1} />
            <text x={m.left - 8} y={y(t)} dy="0.32em" textAnchor="end" className="tnum fill-[var(--muted)] text-[11px]">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        {xLabels.map((l, i) => (
          <text key={l} x={x(i)} y={height - 6} textAnchor="middle" className="tnum fill-[var(--muted)] text-[11px]">
            {l}
          </text>
        ))}
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={m.top} y2={m.top + h} stroke="var(--axis)" strokeWidth={1} />}
        {series.map((s) => (
          <g key={s.name}>
            <path
              d={s.values.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join("")}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {s.values.map((v, i) => (
              <circle key={i} cx={x(i)} cy={y(v)} r={hover === i ? 5 : 4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
            ))}
            {/* direct label at the line end (text in ink, identity from the key beside it) */}
            {labelEnds && <g transform={`translate(${x(s.values.length - 1) + 10},${y(s.values[s.values.length - 1])})`}>
              <line x1={0} x2={8} y1={0} y2={0} stroke={s.color} strokeWidth={2} />
              <text x={12} dy="0.32em" className="fill-[var(--text-2)] text-[11.5px]">
                {s.name} <tspan className="tnum fill-[var(--text)] font-semibold">{Math.round(s.values[s.values.length - 1] * 100)}%</tspan>
              </text>
            </g>}
          </g>
        ))}
      </svg>
      {hover !== null && (
        <Tooltip x={x(hover)} y={Math.min(...series.map((s) => y(s.values[hover])))} width={width}>
          <p className="mb-1 font-medium text-muted">{xLabels[hover]}</p>
          {series.map((s) => (
            <TooltipRow key={s.name} color={s.color} label={s.name} value={`${(s.values[hover] * 100).toFixed(1)}%`} />
          ))}
        </Tooltip>
      )}
    </div>
  );
}
