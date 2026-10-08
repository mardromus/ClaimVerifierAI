import clsx from "clsx";
import { useState, type ReactNode } from "react";

export interface TableData {
  columns: string[];
  rows: (string | number)[][];
}

/** Card wrapper with a Chart / Table toggle: every chart has a table-view twin. */
export function ChartCard({ title, sub, table, children, legend }: { title: string; sub?: string; table: TableData; children: ReactNode; legend?: ReactNode }) {
  const [view, setView] = useState<"chart" | "table">("chart");
  return (
    <section className="card p-5">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-[14.5px] font-semibold tracking-tight">{title}</h3>
          {sub && <p className="mt-0.5 text-[12.5px] text-muted">{sub}</p>}
        </div>
        <div className="flex rounded-lg border border-line bg-surface-2 p-0.5 text-[12px] font-medium" role="tablist" aria-label={`${title} view`}>
          {(["chart", "table"] as const).map((v) => (
            <button
              key={v}
              role="tab"
              aria-selected={view === v}
              onClick={() => setView(v)}
              className={clsx("rounded-md px-2.5 py-1 capitalize", view === v ? "bg-surface text-ink shadow-soft" : "text-muted hover:text-ink")}
            >
              {v}
            </button>
          ))}
        </div>
      </div>
      {view === "chart" ? (
        <>
          {legend && <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1.5 text-[12px] text-ink-2">{legend}</div>}
          {children}
        </>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-line text-left text-muted">
                {table.columns.map((c, i) => (
                  <th key={c} className={clsx("py-2 pr-4 font-medium", i > 0 && "text-right")}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((r, i) => (
                <tr key={i} className="border-b border-line last:border-0">
                  {r.map((v, j) => (
                    <td key={j} className={clsx("py-2 pr-4", j > 0 ? "tnum text-right" : "text-ink-2")}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export function LegendItem({ color, label, shape = "rect" }: { color: string; label: string; shape?: "rect" | "line" }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {shape === "rect" ? (
        <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: color }} />
      ) : (
        <span className="h-[2px] w-3.5 rounded-full" style={{ background: color }} />
      )}
      {label}
    </span>
  );
}

export function Tooltip({ x, y, children, width }: { x: number; y: number; children: ReactNode; width: number }) {
  const left = Math.min(Math.max(x, 70), width - 70);
  return (
    <div
      className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-lg border border-line bg-surface px-2.5 py-1.5 text-[12px] whitespace-nowrap shadow-float"
      style={{ left, top: y - 8 }}
      role="status"
    >
      {children}
    </div>
  );
}

export function TooltipRow({ color, label, value }: { color: string; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="h-[2px] w-3 rounded-full" style={{ background: color }} />
      <span className="tnum font-semibold text-ink">{value}</span>
      <span className="text-muted">{label}</span>
    </div>
  );
}
