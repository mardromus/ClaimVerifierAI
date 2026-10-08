import clsx from "clsx";
import { Check, Download, ExternalLink, FileUp, ListChecks, Play, Square, Wand2, X } from "lucide-react";
import { motion } from "motion/react";
import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Button, Meter, VerdictBadge } from "../components/ui";
import { api } from "../lib/api";
import { parseBatchInput, toCSV, type BatchInput } from "../lib/csv";
import { download } from "../lib/export";
import { pct, truncate } from "../lib/format";
import type { VerificationResult, Verdict } from "../lib/types";
import { VERDICTS, VERDICT_META } from "../lib/verdict";

interface Row extends BatchInput {
  result?: VerificationResult;
  error?: string;
}

const CHUNK = 4;
const MAX = 64;

function topEvidence(r: VerificationResult) {
  const docs = r.documents.filter((d) => d.evidence_weight >= 0.5);
  docs.sort((a, b) => b.evidence_weight * Math.max(b.stance_probs.SUPPORTED, b.stance_probs.CONTRADICTED) - a.evidence_weight * Math.max(a.stance_probs.SUPPORTED, a.stance_probs.CONTRADICTED));
  return docs[0];
}

export default function BatchPage() {
  const [text, setText] = useState("");
  const [rows, setRows] = useState<Row[]>([]);
  const [running, setRunning] = useState(false);
  const [expanded, setExpanded] = useState<number | null>(null);
  const stopRef = useRef(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const parsed = useMemo(() => parseBatchInput(text), [text]);

  const loadExamples = async () => {
    const ex = await api.examples(20, 11);
    setText(toCSV([["claim", "label"], ...ex.map((e) => [e.claim, e.label])]));
  };

  const start = async () => {
    const inputs = parsed.slice(0, MAX);
    setRows(inputs.map((i) => ({ ...i })));
    setRunning(true);
    stopRef.current = false;
    for (let i = 0; i < inputs.length && !stopRef.current; i += CHUNK) {
      const chunk = inputs.slice(i, i + CHUNK);
      try {
        const results = await api.verifyBatch(chunk.map((c) => c.claim));
        setRows((rs) => rs.map((r, j) => (j >= i && j < i + chunk.length ? { ...r, result: results[j - i] } : r)));
      } catch (e) {
        setRows((rs) => rs.map((r, j) => (j >= i && j < i + chunk.length ? { ...r, error: (e as Error).message } : r)));
      }
    }
    setRunning(false);
  };

  const done = rows.filter((r) => r.result || r.error).length;
  const results = rows.filter((r) => r.result);
  const counts = VERDICTS.map((v) => results.filter((r) => r.result!.verdict === v).length);
  const labelled = results.filter((r) => r.gold);
  const correct = labelled.filter((r) => r.result!.verdict === r.gold).length;
  const macroF1 = useMemo(() => {
    if (!labelled.length) return null;
    const f1s = VERDICTS.map((v) => {
      const tp = labelled.filter((r) => r.gold === v && r.result!.verdict === v).length;
      const p = labelled.filter((r) => r.result!.verdict === v).length;
      const g = labelled.filter((r) => r.gold === v).length;
      const prec = p ? tp / p : 0;
      const rec = g ? tp / g : 0;
      return prec + rec ? (2 * prec * rec) / (prec + rec) : 0;
    });
    return f1s.reduce((a, b) => a + b, 0) / f1s.length;
  }, [labelled]);

  const exportCSV = () =>
    download(
      "claimverifier-batch.csv",
      toCSV([
        ["claim", "verdict", "confidence", "p_supported", "p_contradicted", "p_insufficient", "gold", "top_paper", "top_evidence"],
        ...rows.map((r) => {
          const ev = r.result ? topEvidence(r.result) : undefined;
          return [
            r.claim,
            r.result?.verdict ?? (r.error ? "ERROR" : ""),
            r.result ? r.result.confidence.toFixed(4) : "",
            r.result ? r.result.scores.SUPPORTED.toFixed(4) : "",
            r.result ? r.result.scores.CONTRADICTED.toFixed(4) : "",
            r.result ? r.result.scores.INSUFFICIENT_EVIDENCE.toFixed(4) : "",
            r.gold ?? "",
            ev?.title ?? "",
            ev?.evidence[0]?.text ?? "",
          ];
        }),
      ]),
      "text/csv",
    );

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <header className="mb-8">
        <p className="flex items-center gap-2 text-sm font-medium text-accent-ink">
          <ListChecks className="h-4 w-4" /> Batch
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">Verify many claims at once</h1>
        <p className="mt-2 max-w-3xl text-[15px] leading-relaxed text-muted">
          Paste one claim per line, or a CSV with a <code className="kbd">claim</code> column. Add a <code className="kbd">label</code> column
          (supported / contradicted / insufficient) to benchmark the pipeline on your own data. Up to {MAX} claims per run, checked against the
          indexed corpus.
        </p>
      </header>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="card p-4">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={8}
            aria-label="Claims to verify"
            placeholder={"Aspirin reduces the risk of colorectal cancer.\nVitamin D supplementation prevents fractures in older adults.\n…"}
            className="scrollbar-thin w-full resize-y rounded-xl bg-surface-2 p-3.5 font-mono text-[13px] leading-relaxed outline-none placeholder:text-muted"
          />
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <input
              ref={fileRef}
              type="file"
              accept=".txt,.csv,text/plain,text/csv"
              className="hidden"
              onChange={async (e) => {
                const f = e.target.files?.[0];
                if (f) setText(await f.text());
                e.target.value = "";
              }}
            />
            <Button size="sm" icon={<FileUp className="h-3.5 w-3.5" />} onClick={() => fileRef.current?.click()}>Upload .txt / .csv</Button>
            <Button size="sm" icon={<Wand2 className="h-3.5 w-3.5" />} onClick={loadExamples}>Load 20 labelled SciFact claims</Button>
            <span className="ml-auto text-xs text-muted">
              {parsed.length} claim{parsed.length === 1 ? "" : "s"}
              {parsed.some((p) => p.gold) && ` · ${parsed.filter((p) => p.gold).length} labelled`}
              {parsed.length > MAX && ` · first ${MAX} will run`}
            </span>
            {running ? (
              <Button variant="secondary" icon={<Square className="h-3.5 w-3.5 fill-current" />} onClick={() => (stopRef.current = true)}>Stop</Button>
            ) : (
              <Button variant="primary" icon={<Play className="h-3.5 w-3.5 fill-current" />} disabled={!parsed.length} onClick={start}>Run batch</Button>
            )}
          </div>
        </div>

        <div className="card p-4">
          <h2 className="text-[12px] font-semibold tracking-wide text-muted uppercase">Summary</h2>
          {rows.length === 0 ? (
            <p className="mt-3 text-[13px] text-muted">Results appear here as claims are verified.</p>
          ) : (
            <>
              <div className="mt-3 mb-1 flex justify-between text-xs text-muted">
                <span>{running ? "Verifying…" : "Finished"}</span>
                <span className="tnum">{done} / {rows.length}</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-surface-3">
                <motion.div className="h-full bg-accent" animate={{ width: `${(100 * done) / rows.length}%` }} />
              </div>
              {results.length > 0 && (
                <>
                  <div className="mt-4 flex h-2.5 gap-[2px] overflow-hidden rounded-full">
                    {VERDICTS.map((v, i) => (
                      <span key={v} style={{ width: `${(100 * counts[i]) / results.length}%`, background: VERDICT_META[v].color }} className="h-full first:rounded-l-full last:rounded-r-full" />
                    ))}
                  </div>
                  <ul className="mt-2.5 space-y-1 text-[12.5px]">
                    {VERDICTS.map((v, i) => {
                      const Icon = VERDICT_META[v].icon;
                      return (
                        <li key={v} className="flex items-center gap-2">
                          <Icon className="h-3.5 w-3.5" style={{ color: VERDICT_META[v].color }} />
                          <span className="text-ink-2">{VERDICT_META[v].label}</span>
                          <span className="tnum ml-auto font-medium">{counts[i]}</span>
                        </li>
                      );
                    })}
                  </ul>
                </>
              )}
              {labelled.length > 0 && (
                <div className="mt-4 grid grid-cols-2 gap-2 border-t border-line pt-4">
                  <div>
                    <p className="text-xs text-muted">Accuracy vs gold</p>
                    <p className="text-xl font-semibold">{pct(correct / labelled.length, 1)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted">Macro F1</p>
                    <p className="text-xl font-semibold">{macroF1 !== null ? (macroF1 * 100).toFixed(1) : "–"}</p>
                  </div>
                  <p className="col-span-2 text-[11.5px] text-muted">{labelled.length} labelled claims scored</p>
                </div>
              )}
              {results.length > 0 && !running && (
                <Button className="mt-4 w-full" size="sm" icon={<Download className="h-3.5 w-3.5" />} onClick={exportCSV}>Export CSV</Button>
              )}
            </>
          )}
        </div>
      </div>

      {rows.length > 0 && (
        <div className="card mt-5 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-[13px]">
              <thead>
                <tr className="border-b border-line bg-surface-2/60 text-left text-[12px] text-muted">
                  <th className="w-10 px-4 py-2.5 font-medium">#</th>
                  <th className="px-2 py-2.5 font-medium">Claim</th>
                  <th className="px-2 py-2.5 font-medium">Verdict</th>
                  <th className="px-2 py-2.5 font-medium">Confidence</th>
                  {rows.some((r) => r.gold) && <th className="px-2 py-2.5 font-medium">Gold</th>}
                  <th className="w-10 px-4 py-2.5" />
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const ev = r.result ? topEvidence(r.result) : undefined;
                  const ok = r.gold && r.result ? r.gold === r.result.verdict : null;
                  return (
                    <FragmentRow
                      key={i}
                      i={i}
                      row={r}
                      ok={ok}
                      showGold={rows.some((x) => x.gold)}
                      expanded={expanded === i}
                      onToggle={() => setExpanded(expanded === i ? null : i)}
                      evidenceTitle={ev?.title}
                      evidenceText={ev?.evidence[0]?.text}
                      evidenceStance={ev?.stance}
                    />
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function FragmentRow({
  i, row, ok, showGold, expanded, onToggle, evidenceTitle, evidenceText, evidenceStance,
}: {
  i: number; row: Row; ok: boolean | null; showGold: boolean; expanded: boolean; onToggle: () => void;
  evidenceTitle?: string; evidenceText?: string; evidenceStance?: Verdict;
}) {
  const r = row.result;
  return (
    <>
      <tr className={clsx("border-b border-line transition-colors", r && "cursor-pointer hover:bg-surface-2/60")} onClick={r ? onToggle : undefined}>
        <td className="tnum px-4 py-2.5 text-muted">{i + 1}</td>
        <td className="px-2 py-2.5 text-ink-2">{truncate(row.claim, 120)}</td>
        <td className="px-2 py-2.5">
          {r ? <VerdictBadge verdict={r.verdict} size="sm" /> : row.error ? <span className="text-xs text-con-ink">{truncate(row.error, 40)}</span> : <span className="shimmer inline-block h-5 w-24 rounded-full" />}
        </td>
        <td className="px-2 py-2.5">
          {r && (
            <span className="inline-flex items-center gap-2">
              <Meter value={r.confidence} color={VERDICT_META[r.verdict].color} />
              <span className="tnum text-xs">{pct(r.confidence)}</span>
            </span>
          )}
        </td>
        {showGold && (
          <td className="px-2 py-2.5 text-xs">
            {row.gold && (
              <span className="inline-flex items-center gap-1 text-muted">
                {ok === true && <Check className="h-3.5 w-3.5 text-sup" aria-label="correct" />}
                {ok === false && <X className="h-3.5 w-3.5 text-con" aria-label="wrong" />}
                {VERDICT_META[row.gold].short}
              </span>
            )}
          </td>
        )}
        <td className="px-4 py-2.5 text-right">
          {r && (
            <Link to={`/?q=${encodeURIComponent(row.claim)}`} onClick={(e) => e.stopPropagation()} className="text-muted hover:text-accent-ink" aria-label="Open in verifier">
              <ExternalLink className="h-3.5 w-3.5" />
            </Link>
          )}
        </td>
      </tr>
      {expanded && r && (
        <tr className="border-b border-line bg-surface-2/40">
          <td />
          <td colSpan={showGold ? 5 : 4} className="px-2 py-3">
            {evidenceTitle ? (
              <div className="rounded-xl border-l-[3px] px-3.5 py-2.5" style={{ borderColor: VERDICT_META[evidenceStance!].color, background: VERDICT_META[evidenceStance!].soft }}>
                <p className="text-[13px] leading-relaxed text-ink">{evidenceText}</p>
                <p className="mt-1 text-[11.5px] text-muted">{evidenceTitle}</p>
              </div>
            ) : (
              <p className="text-[12.5px] text-muted">No retrieved paper was relevant enough to count as evidence.</p>
            )}
          </td>
        </tr>
      )}
    </>
  );
}
