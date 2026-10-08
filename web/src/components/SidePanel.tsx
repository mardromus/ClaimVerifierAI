import { Braces, Check, Copy, Cpu, FileDown, Link2, RotateCcw, Timer } from "lucide-react";
import { useState, type ReactNode } from "react";
import { download, resultToMarkdown, slug } from "../lib/export";
import { ms, truncate } from "../lib/format";
import type { VerificationResult } from "../lib/types";
import { VERDICT_META } from "../lib/verdict";
import { useCitations } from "./citationContext";
import { isShown } from "./PaperCard";
import { useToast } from "./ui";

function Panel({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <section className="card p-4">
      <h3 className="mb-3 flex items-center gap-2 text-[12px] font-semibold tracking-wide text-muted uppercase">
        {icon}
        {title}
      </h3>
      {children}
    </section>
  );
}

function ActionButton({ icon, label, onClick }: { icon: ReactNode; label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-[13px] font-medium text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink"
    >
      {icon}
      {label}
    </button>
  );
}

export function SidePanel({ result, onRerun }: { result: VerificationResult; onRerun: () => void }) {
  const toast = useToast();
  const { setActive, focus } = useCitations();
  const [copied, setCopied] = useState(false);
  const copy = async (text: string, msg: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast(msg);
    } catch {
      toast("Copy failed: clipboard unavailable");
    }
  };
  const shareUrl = () => {
    const u = new URL(window.location.href);
    u.pathname = "/";
    u.search = new URLSearchParams({ q: result.claim, source: result.source === "custom" ? "corpus" : result.source }).toString();
    return u.toString();
  };
  const timings = Object.entries(result.timings_ms ?? {});
  const comps = result.components ?? {};
  const models: [string, string][] = [
    ["Retrieval", `${comps.retrieval?.embedder === "sbert" ? String(comps.retrieval?.model_name ?? "Sentence-BERT").split("/").pop() : "TF-IDF/LSA"} + FAISS${comps.retrieval?.hybrid_bm25 ? " + BM25" : ""}`],
    ["Evidence", comps.rationale?.method === "scibert" ? "SciBERT (fine-tuned)" : String(comps.rationale?.method ?? "")],
    ["Stance", comps.nli?.method === "transformer" ? String(comps.nli?.model ?? "").split("/").pop() ?? "" : "lexical stance model"],
  ];
  return (
    <div className="space-y-4">
      <Panel title="Sources" icon={<Link2 className="h-3.5 w-3.5" />}>
        <ul className="space-y-1">
          {result.documents.map((d) => {
            const shown = isShown(d);
            const color = shown ? VERDICT_META[d.stance].color : "var(--surface-3)";
            return (
              <li key={`${d.doc_id}-${d.citation}`}>
                <button
                  className="flex w-full items-start gap-2 rounded-lg px-1.5 py-1.5 text-left hover:bg-surface-2"
                  onMouseEnter={() => setActive(d.citation)}
                  onMouseLeave={() => setActive(null)}
                  onClick={() => focus(d.citation)}
                >
                  <span className="tnum mt-px w-4 shrink-0 text-right text-[11.5px] font-semibold text-muted">{d.citation}</span>
                  <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
                  <span className="text-[12.5px] leading-snug text-ink-2">{truncate(d.title, 80)}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </Panel>
      <Panel title="Actions" icon={<FileDown className="h-3.5 w-3.5" />}>
        <div className="-mx-1">
          <ActionButton
            icon={copied ? <Check className="h-4 w-4 text-sup" /> : <Link2 className="h-4 w-4 text-muted" />}
            label="Copy share link"
            onClick={() => {
              copy(shareUrl(), "Link copied");
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            }}
          />
          <ActionButton icon={<Copy className="h-4 w-4 text-muted" />} label="Copy as Markdown" onClick={() => copy(resultToMarkdown(result), "Markdown copied")} />
          <ActionButton
            icon={<FileDown className="h-4 w-4 text-muted" />}
            label="Download report (.md)"
            onClick={() => download(`claim-${slug(result.claim)}.md`, resultToMarkdown(result), "text/markdown")}
          />
          <ActionButton
            icon={<Braces className="h-4 w-4 text-muted" />}
            label="Download JSON"
            onClick={() => download(`claim-${slug(result.claim)}.json`, JSON.stringify(result, null, 2), "application/json")}
          />
          <ActionButton icon={<RotateCcw className="h-4 w-4 text-muted" />} label="Run again" onClick={onRerun} />
        </div>
      </Panel>
      <Panel title="Models" icon={<Cpu className="h-3.5 w-3.5" />}>
        <dl className="space-y-1.5 text-[12.5px]">
          {models.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-3">
              <dt className="text-muted">{k}</dt>
              <dd className="truncate text-right font-medium text-ink-2" title={v}>{v}</dd>
            </div>
          ))}
          {comps.decision && (
            <div className="flex justify-between gap-3">
              <dt className="text-muted">Threshold</dt>
              <dd className="tnum font-medium text-ink-2">
                {Number(comps.decision.threshold).toFixed(2)}
                {comps.decision.calibrated ? " (calibrated)" : ""}
              </dd>
            </div>
          )}
        </dl>
      </Panel>
      {timings.length > 0 && (
        <Panel title="Latency" icon={<Timer className="h-3.5 w-3.5" />}>
          <dl className="space-y-1.5 text-[12.5px]">
            {timings.map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <dt className="text-muted capitalize">{k === "nli" ? "NLI" : k}</dt>
                <dd className="tnum font-medium text-ink-2">{ms(v)}</dd>
              </div>
            ))}
          </dl>
        </Panel>
      )}
    </div>
  );
}
