import clsx from "clsx";
import { FlaskConical } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ChartCard, LegendItem } from "../components/charts/ChartCard";
import { ConfusionMatrix } from "../components/charts/ConfusionMatrix";
import { GroupedBars } from "../components/charts/GroupedBars";
import { LineChart } from "../components/charts/LineChart";
import { StatTile } from "../components/charts/StatTile";
import { api } from "../lib/api";
import { pct } from "../lib/format";
import { MAJORITY, reportLabel, SERIES } from "../lib/reports";
import type { ReportMetrics, ReportSummary } from "../lib/types";
import { VERDICTS, VERDICT_META } from "../lib/verdict";

const KS = [1, 3, 5, 10, 20];

export default function EvaluationPage() {
  const [reports, setReports] = useState<ReportSummary[] | null>(null);
  const [metrics, setMetrics] = useState<Record<string, ReportMetrics>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .reports()
      .then((rs) => {
        const sorted = [...rs].sort((a, b) => a.macro_f1 - b.macro_f1 || a.name.localeCompare(b.name));
        setReports(sorted);
        const best = [...rs].sort((a, b) => b.macro_f1 - a.macro_f1)[0];
        setSelected(best?.name ?? null);
        rs.forEach((r) => api.report(r.name).then((d) => setMetrics((m) => ({ ...m, [r.name]: d.metrics }))));
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const charted = useMemo(() => (reports ?? []).slice(0, 4), [reports]);
  const colorOf = (name: string) => SERIES[charted.findIndex((r) => r.name === name)] ?? "var(--muted)";
  const sel = selected ? metrics[selected] : undefined;
  const selSummary = reports?.find((r) => r.name === selected);

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <header className="mb-8">
        <p className="flex items-center gap-2 text-sm font-medium text-accent-ink">
          <FlaskConical className="h-4 w-4" /> Evaluation
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">How well does it work?</h1>
        <p className="mt-2 max-w-3xl text-[15px] leading-relaxed text-muted">
          Every pipeline is scored on the SciFact <b className="font-medium text-ink-2">dev split</b> (300 expert-written claims), which is never
          used for training or tuning. Claim-level accuracy and macro-F1 measure the verdict; Recall@K and MRR measure whether the right papers
          are found; abstract-level F1 is the official SciFact metric.
        </p>
      </header>

      {error && <p className="card p-5 text-sm text-con-ink">Could not load reports: {error}</p>}
      {reports && reports.length === 0 && (
        <p className="card p-6 text-sm text-muted">
          No reports yet. Run <code className="kbd">python -m claimverifier evaluate --config configs/scibert.yaml</code> to create one.
        </p>
      )}

      {reports && reports.length > 0 && (
        <>
          <div className="mb-6 flex flex-wrap gap-2" role="radiogroup" aria-label="Report">
            {reports.map((r) => (
              <button
                key={r.name}
                role="radio"
                aria-checked={selected === r.name}
                onClick={() => setSelected(r.name)}
                className={clsx(
                  "inline-flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-[13px] font-medium transition-colors",
                  selected === r.name ? "border-accent/40 bg-accent-soft text-accent-ink" : "border-line bg-surface text-muted hover:text-ink",
                )}
              >
                <span className="h-2 w-2 rounded-full" style={{ background: colorOf(r.name) }} />
                {reportLabel(r.name)}
                <span className="tnum text-[11.5px] opacity-75">F1 {(r.macro_f1 * 100).toFixed(1)}</span>
              </button>
            ))}
          </div>

          {selSummary && (
            <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
              <StatTile label="Verdict accuracy" value={pct(selSummary.accuracy, 1)} sub={`Majority baseline ${pct(MAJORITY.accuracy, 1)}`} />
              <StatTile label="Macro F1" value={(selSummary.macro_f1 * 100).toFixed(1)} sub={`Majority baseline ${(MAJORITY.macro_f1 * 100).toFixed(1)}`} />
              <StatTile label="Evidence Recall@5" value={selSummary.recall_at_5 != null ? pct(selSummary.recall_at_5, 1) : "–"} sub="Gold papers in the top 5" />
              <StatTile label="Mean reciprocal rank" value={selSummary.mrr != null ? selSummary.mrr.toFixed(3) : "–"} sub="First gold paper's rank" />
            </div>
          )}

          <div className="grid gap-5 lg:grid-cols-2">
            <ChartCard
              title="Pipelines compared"
              sub="Claim-level accuracy and macro-F1, SciFact abstract-level label F1"
              legend={charted.map((r) => <LegendItem key={r.name} color={colorOf(r.name)} label={reportLabel(r.name)} />)}
              table={{
                columns: ["Pipeline", "Accuracy", "Macro P", "Macro R", "Macro F1", "Abstract F1", "R@5", "MRR"],
                rows: reports.map((r) => [
                  reportLabel(r.name), pct(r.accuracy, 1), pct(r.macro_precision, 1), pct(r.macro_recall, 1), pct(r.macro_f1, 1),
                  pct(r.abstract_label_f1, 1), r.recall_at_5 != null ? pct(r.recall_at_5, 1) : "–", r.mrr != null ? r.mrr.toFixed(3) : "–",
                ]),
              }}
            >
              <GroupedBars
                categories={["Accuracy", "Macro F1", "Abstract F1"]}
                series={charted.map((r) => ({ name: reportLabel(r.name), color: colorOf(r.name), values: [r.accuracy, r.macro_f1, r.abstract_label_f1] }))}
              />
            </ChartCard>

            <ChartCard
              title="F1 by verdict class"
              sub="Contradictions are the hardest class for every pipeline"
              legend={charted.map((r) => <LegendItem key={r.name} color={colorOf(r.name)} label={reportLabel(r.name)} />)}
              table={{
                columns: ["Pipeline", ...VERDICTS.map((v) => `${VERDICT_META[v].short} F1`)],
                rows: charted.map((r) => [
                  reportLabel(r.name),
                  ...VERDICTS.map((v) => (metrics[r.name] ? pct(metrics[r.name].verdict_classification.per_class[v].f1, 1) : "…")),
                ]),
              }}
            >
              <GroupedBars
                categories={VERDICTS.map((v) => VERDICT_META[v].short)}
                series={charted
                  .filter((r) => metrics[r.name])
                  .map((r) => ({
                    name: reportLabel(r.name),
                    color: colorOf(r.name),
                    values: VERDICTS.map((v) => metrics[r.name].verdict_classification.per_class[v].f1),
                  }))}
              />
            </ChartCard>

            {sel && (
              <ChartCard
                title="Evidence retrieval"
                sub={`Recall@K and hit rate over ${sel.evidence_retrieval.num_queries} claims with gold evidence · ${reportLabel(selected!)}`}
                legend={[
                  <LegendItem key="r" color="var(--s1)" label="Recall@K" shape="line" />,
                  <LegendItem key="h" color="var(--s2)" label="Hit rate@K" shape="line" />,
                ]}
                table={{
                  columns: ["K", "Recall", "Hit rate", "Precision"],
                  rows: KS.map((k) => [
                    `@${k}`, pct(sel.evidence_retrieval[`recall@${k}`], 1), pct(sel.evidence_retrieval[`hit@${k}`], 1), pct(sel.evidence_retrieval[`precision@${k}`], 1),
                  ]),
                }}
              >
                <LineChart
                  xLabels={KS.map((k) => `@${k}`)}
                  series={[
                    { name: "Recall", color: "var(--s1)", values: KS.map((k) => sel.evidence_retrieval[`recall@${k}`]) },
                    { name: "Hit rate", color: "var(--s2)", values: KS.map((k) => sel.evidence_retrieval[`hit@${k}`]) },
                  ]}
                />
              </ChartCard>
            )}

            {sel && (
              <ChartCard
                title="Confusion matrix"
                sub={`${reportLabel(selected!)} · rows are gold labels`}
                table={{
                  columns: ["Gold \\ predicted", ...VERDICTS.map((v) => VERDICT_META[v].short)],
                  rows: sel.verdict_classification.confusion_matrix.matrix.map((row, i) => [VERDICT_META[VERDICTS[i]].short, ...row]),
                }}
              >
                <ConfusionMatrix labels={VERDICTS.map((v) => VERDICT_META[v].short)} matrix={sel.verdict_classification.confusion_matrix.matrix} />
              </ChartCard>
            )}
          </div>

          {sel && (
            <div className="mt-5 grid gap-5 lg:grid-cols-2">
              <section className="card overflow-x-auto p-5">
                <h3 className="mb-3 text-[14.5px] font-semibold tracking-tight">Per-class precision / recall</h3>
                <table className="w-full text-[13px]">
                  <thead>
                    <tr className="border-b border-line text-left text-muted">
                      <th className="py-2 font-medium">Class</th>
                      <th className="py-2 text-right font-medium">Precision</th>
                      <th className="py-2 text-right font-medium">Recall</th>
                      <th className="py-2 text-right font-medium">F1</th>
                      <th className="py-2 text-right font-medium">Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {VERDICTS.map((v) => {
                      const c = sel.verdict_classification.per_class[v];
                      const Icon = VERDICT_META[v].icon;
                      return (
                        <tr key={v} className="border-b border-line last:border-0">
                          <td className="py-2">
                            <span className="inline-flex items-center gap-1.5">
                              <Icon className="h-3.5 w-3.5" style={{ color: VERDICT_META[v].color }} />
                              {VERDICT_META[v].label}
                            </span>
                          </td>
                          <td className="tnum py-2 text-right">{pct(c.precision, 1)}</td>
                          <td className="tnum py-2 text-right">{pct(c.recall, 1)}</td>
                          <td className="tnum py-2 text-right font-semibold">{pct(c.f1, 1)}</td>
                          <td className="tnum py-2 text-right text-muted">{c.support}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </section>
              <section className="card overflow-x-auto p-5">
                <h3 className="mb-3 text-[14.5px] font-semibold tracking-tight">SciFact abstract-level evaluation</h3>
                <table className="w-full text-[13px]">
                  <thead>
                    <tr className="border-b border-line text-left text-muted">
                      <th className="py-2 font-medium" />
                      <th className="py-2 text-right font-medium">Precision</th>
                      <th className="py-2 text-right font-medium">Recall</th>
                      <th className="py-2 text-right font-medium">F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(
                      [
                        ["Label only", sel.scifact_abstract_level.label_only],
                        ["Rationalized", sel.scifact_abstract_level.rationalized],
                        ["Sentence selection", sel.scifact_sentence_selection],
                      ] as const
                    ).map(([k, v]) => (
                      <tr key={k} className="border-b border-line last:border-0">
                        <td className="py-2 text-ink-2">{k}</td>
                        <td className="tnum py-2 text-right">{pct(v.precision, 1)}</td>
                        <td className="tnum py-2 text-right">{pct(v.recall, 1)}</td>
                        <td className="tnum py-2 text-right font-semibold">{pct(v.f1, 1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-3 text-[12px] leading-relaxed text-muted">
                  An abstract counts as correct when it is gold evidence and its predicted label matches; “rationalized” also requires the
                  highlighted sentences to cover a full gold rationale. Runs at {Math.round(sel.ms_per_claim)} ms per claim on CPU.
                </p>
              </section>
            </div>
          )}
        </>
      )}
    </div>
  );
}
