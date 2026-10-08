import clsx from "clsx";
import { ArrowUp, BookOpen, Database, FileText, Globe, Plus, SlidersHorizontal, Square, Trash2 } from "lucide-react";
import { useEffect, useRef, type KeyboardEvent } from "react";
import type { CustomDocument, SourceInfo, SourceName } from "../lib/types";
import { Popover, Switch } from "./ui";

const SOURCE_ICONS: Record<SourceName, typeof Database> = { corpus: Database, europepmc: Globe, pubmed: BookOpen, custom: FileText };
const SHORT_LABELS: Record<SourceName, string> = { corpus: "SciFact", europepmc: "Europe PMC", pubmed: "PubMed", custom: "Your text" };

export interface InputSettings {
  topK: number;
  explain: boolean;
}

interface Props {
  claim: string;
  onClaim: (v: string) => void;
  source: SourceName;
  onSource: (s: SourceName) => void;
  sources: SourceInfo[];
  settings: InputSettings;
  onSettings: (s: InputSettings) => void;
  documents: CustomDocument[];
  onDocuments: (d: CustomDocument[]) => void;
  running: boolean;
  onSubmit: () => void;
  onCancel: () => void;
  compact?: boolean;
  autoFocus?: boolean;
}

export function ClaimInput(p: Props) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [p.claim]);
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        ref.current?.focus();
        ref.current?.select();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const canSubmit = p.claim.trim().length >= 3 && (p.source !== "custom" || p.documents.some((d) => d.abstract.trim()));
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      if (canSubmit && !p.running) p.onSubmit();
    }
  };
  const sources = p.sources.length ? p.sources : [{ name: "corpus" as SourceName, label: "SciFact corpus", description: "" }];

  return (
    <div className={clsx("card overflow-hidden transition-shadow focus-within:shadow-float", p.compact ? "" : "shadow-float")}>
      <div className="flex items-start gap-3 px-4 pt-4 sm:px-5">
        <textarea
          ref={ref}
          value={p.claim}
          onChange={(e) => p.onClaim(e.target.value)}
          onKeyDown={onKeyDown}
          rows={p.compact ? 1 : 2}
          maxLength={1000}
          autoFocus={p.autoFocus}
          aria-label="Scientific claim"
          placeholder="Enter a scientific claim, e.g. “Vitamin D supplementation reduces the risk of fractures in older adults.”"
          className={clsx(
            "w-full resize-none bg-transparent leading-relaxed text-ink outline-none placeholder:text-muted/80",
            p.compact ? "text-[15px]" : "text-[17px] sm:text-lg",
          )}
        />
      </div>

      {p.source === "custom" && <CustomDocsEditor documents={p.documents} onChange={p.onDocuments} />}

      <div className="flex flex-wrap items-center gap-2 px-3 pt-3 pb-3 sm:px-4">
        <div className="scrollbar-thin -mx-1 flex max-w-full items-center gap-1 overflow-x-auto px-1" role="radiogroup" aria-label="Evidence source">
          {sources.map((s) => {
            const Icon = SOURCE_ICONS[s.name] ?? Database;
            const active = p.source === s.name;
            return (
              <button
                key={s.name}
                type="button"
                role="radio"
                aria-checked={active}
                title={`${s.label}: ${s.description}`}
                onClick={() => p.onSource(s.name)}
                className={clsx(
                  "inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full border px-3 text-[12.5px] font-medium transition-colors",
                  active ? "border-accent/40 bg-accent-soft text-accent-ink" : "border-line text-muted hover:bg-surface-2 hover:text-ink",
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                {SHORT_LABELS[s.name] ?? s.label}
                {(s.name === "europepmc" || s.name === "pubmed") && (
                  <span className={clsx("rounded px-1 text-[10px] font-semibold uppercase tracking-wide", active ? "bg-accent/15" : "bg-surface-3")}>live</span>
                )}
              </button>
            );
          })}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Popover
            align="right"
            trigger={(open) => (
              <button
                type="button"
                aria-label="Settings"
                className={clsx(
                  "inline-flex h-8 items-center gap-1.5 rounded-full border border-line px-3 text-[12.5px] font-medium transition-colors",
                  open ? "bg-surface-2 text-ink" : "text-muted hover:text-ink",
                )}
              >
                <SlidersHorizontal className="h-3.5 w-3.5" />
                <span className="tnum">top {p.settings.topK}</span>
              </button>
            )}
          >
            <div className="space-y-4 text-sm">
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <label htmlFor="topk" className="font-medium">Papers to analyse</label>
                  <span className="tnum rounded-md bg-surface-2 px-1.5 text-[13px]">{p.settings.topK}</span>
                </div>
                <input
                  id="topk"
                  type="range"
                  min={1}
                  max={10}
                  value={p.settings.topK}
                  onChange={(e) => p.onSettings({ ...p.settings, topK: Number(e.target.value) })}
                  className="w-full accent-[var(--accent)]"
                />
                <p className="mt-1 text-xs text-muted">Abstracts whose sentences are checked for evidence.</p>
              </div>
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="font-medium">Explanation</p>
                  <p className="text-xs text-muted">LLM summary with citations</p>
                </div>
                <Switch label="Generate explanation" checked={p.settings.explain} onChange={(explain) => p.onSettings({ ...p.settings, explain })} />
              </div>
            </div>
          </Popover>
          <span className="hidden text-xs text-muted lg:inline">
            <kbd className="kbd">↵</kbd> to verify
          </span>
          {p.running ? (
            <button
              type="button"
              onClick={p.onCancel}
              aria-label="Stop"
              className="grid h-9 w-9 place-items-center rounded-full bg-ink text-bg transition-transform hover:scale-105"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
            </button>
          ) : (
            <button
              type="button"
              onClick={p.onSubmit}
              disabled={!canSubmit}
              aria-label="Verify claim"
              className="grid h-9 w-9 place-items-center rounded-full bg-accent text-white shadow-soft transition-[transform,opacity] hover:scale-105 disabled:scale-100 disabled:opacity-35"
            >
              <ArrowUp className="h-4.5 w-4.5" strokeWidth={2.5} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function CustomDocsEditor({ documents, onChange }: { documents: CustomDocument[]; onChange: (d: CustomDocument[]) => void }) {
  const update = (i: number, patch: Partial<CustomDocument>) => onChange(documents.map((d, j) => (j === i ? { ...d, ...patch } : d)));
  return (
    <div className="mx-4 mt-3 space-y-2 sm:mx-5">
      {documents.map((d, i) => (
        <div key={i} className="rounded-xl border border-line bg-surface-2/60 p-3">
          <div className="mb-1.5 flex items-center gap-2">
            <span className="grid h-5 w-5 place-items-center rounded-md bg-surface-3 text-[11px] font-semibold text-muted">{i + 1}</span>
            <input
              value={d.title}
              onChange={(e) => update(i, { title: e.target.value })}
              placeholder="Title (optional)"
              aria-label={`Title of abstract ${i + 1}`}
              className="min-w-0 flex-1 bg-transparent text-[13.5px] font-medium outline-none placeholder:text-muted"
            />
            {documents.length > 1 && (
              <button aria-label={`Remove abstract ${i + 1}`} onClick={() => onChange(documents.filter((_, j) => j !== i))} className="text-muted hover:text-con-ink">
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
          <textarea
            value={d.abstract}
            onChange={(e) => update(i, { abstract: e.target.value })}
            rows={4}
            placeholder="Paste an abstract or a passage from a paper…"
            aria-label={`Text of abstract ${i + 1}`}
            className="w-full resize-y bg-transparent text-[13.5px] leading-relaxed outline-none placeholder:text-muted"
          />
        </div>
      ))}
      {documents.length < 5 && (
        <button
          onClick={() => onChange([...documents, { title: "", abstract: "" }])}
          className="inline-flex items-center gap-1 text-[12.5px] font-medium text-accent-ink hover:underline"
        >
          <Plus className="h-3.5 w-3.5" /> Add another abstract
        </button>
      )}
    </div>
  );
}
