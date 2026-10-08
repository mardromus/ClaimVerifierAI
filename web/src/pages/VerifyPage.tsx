import { AlertCircle, FileSearch, Gauge, Quote, ScanText, Sparkles } from "lucide-react";
import { motion } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { CandidateList } from "../components/CandidateList";
import { ClaimInput, type InputSettings } from "../components/ClaimInput";
import { CitationProvider } from "../components/citationContext";
import { EvidenceSection } from "../components/EvidenceSection";
import { ExampleChips } from "../components/ExampleChips";
import { ExplanationCard } from "../components/ExplanationCard";
import { PipelineProgress } from "../components/PipelineProgress";
import { SidePanel } from "../components/SidePanel";
import { Button } from "../components/ui";
import { VerdictCard } from "../components/VerdictCard";
import { useApp } from "../hooks/useApp";
import { useVerifyStream } from "../hooks/useVerifyStream";
import { compact } from "../lib/format";
import type { CustomDocument, SourceName } from "../lib/types";

const SOURCES: SourceName[] = ["corpus", "europepmc", "pubmed", "custom"];
const SETTINGS_KEY = "claimverifier.settings.v1";

function loadSettings(): InputSettings {
  try {
    const s = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}");
    return { topK: Number(s.topK) || 5, explain: s.explain !== false };
  } catch {
    return { topK: 5, explain: true };
  }
}

const FEATURES = [
  { icon: FileSearch, title: "Finds the papers", text: "Hybrid semantic + BM25 search over research abstracts, or live Europe PMC and PubMed." },
  { icon: ScanText, title: "Pinpoints evidence", text: "A fine-tuned SciBERT reads every sentence and highlights the ones that matter." },
  { icon: Gauge, title: "Weighs the stance", text: "NLI decides whether each paper supports or contradicts the claim, with calibrated confidence." },
  { icon: Quote, title: "Explains with citations", text: "An LLM writes a short explanation grounded only in the evidence, citing every source." },
];

export default function VerifyPage() {
  const { info, history, remember } = useApp();
  const [params, setParams] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [claim, setClaim] = useState(params.get("q") ?? "");
  const [source, setSource] = useState<SourceName>(() => {
    const s = params.get("source") as SourceName | null;
    return s && SOURCES.includes(s) && s !== "custom" ? s : "corpus";
  });
  const [settings, setSettingsState] = useState<InputSettings>(loadSettings);
  const [documents, setDocuments] = useState<CustomDocument[]>([{ title: "", abstract: "" }]);
  const { state, run, cancel, show, reset } = useVerifyStream(remember);
  const autoRan = useRef(false);

  const setSettings = (s: InputSettings) => {
    setSettingsState(s);
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
    } catch {
      /* storage unavailable */
    }
  };

  const submit = useCallback(
    (text = claim, src = source) => {
      const c = text.trim();
      if (c.length < 3) return;
      run({
        claim: c,
        top_k: settings.topK,
        explain: settings.explain,
        source: src,
        documents: src === "custom" ? documents.filter((d) => d.abstract.trim()) : undefined,
      });
      if (src !== "custom") setParams({ q: c, source: src }, { replace: true });
      else setParams({}, { replace: true });
      window.scrollTo({ top: 0, behavior: "smooth" });
    },
    [claim, source, settings, documents, run, setParams],
  );

  // Open a result from history, or run a claim passed in the URL (?q=...&source=...).
  useEffect(() => {
    const historyId = (location.state as { historyId?: string } | null)?.historyId;
    if (historyId) {
      const entry = history.find((h) => h.id === historyId);
      navigate(".", { replace: true, state: null });
      if (entry) {
        setClaim(entry.claim);
        if (entry.result) show(entry.result);
        else submit(entry.claim, entry.source === "custom" ? "corpus" : entry.source);
      }
      return;
    }
    if (!autoRan.current && params.get("q")) {
      autoRan.current = true;
      submit(params.get("q") ?? "", source);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.state]);

  const idle = state.status === "idle";
  const input = (
    <ClaimInput
      claim={claim}
      onClaim={setClaim}
      source={source}
      onSource={setSource}
      sources={info?.sources ?? []}
      settings={settings}
      onSettings={setSettings}
      documents={documents}
      onDocuments={setDocuments}
      running={state.status === "running"}
      onSubmit={() => submit()}
      onCancel={cancel}
      compact={!idle}
      autoFocus={idle}
    />
  );

  if (idle) {
    return (
      <div className="relative overflow-hidden">
        <div className="bg-dots pointer-events-none absolute inset-0 [mask-image:radial-gradient(ellipse_70%_60%_at_50%_0%,black,transparent)]" />
        <div className="bg-glow pointer-events-none absolute inset-0" />
        <div className="relative mx-auto max-w-3xl px-4 pt-14 pb-10 sm:pt-24">
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="text-center">
            <span className="inline-flex items-center gap-2 rounded-full border border-line bg-surface/80 px-3 py-1 text-xs font-medium text-muted shadow-soft backdrop-blur">
              <Sparkles className="h-3.5 w-3.5 text-accent" />
              {info ? `${compact(info.stats.documents)} indexed abstracts` : "SciFact"} · SciBERT · DeBERTa-v3 · Llama 3 / Qwen 2.5
            </span>
            <h1 className="mt-5 text-[40px] leading-[1.05] font-semibold tracking-[-0.035em] text-balance sm:text-6xl">
              Check any scientific claim{" "}
              <span className="bg-gradient-to-r from-[var(--accent)] via-[var(--nei)] to-[var(--sup)] bg-clip-text text-transparent">against the evidence</span>
            </h1>
            <p className="mx-auto mt-4 max-w-xl text-[15.5px] leading-relaxed text-pretty text-muted sm:text-[17px]">
              ClaimVerifier reads the research literature for you: it retrieves papers, highlights the sentences that matter, decides whether
              they support or contradict the claim, and explains its verdict with citations.
            </p>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, delay: 0.08 }} className="mt-8">
            {input}
            <ExampleChips onPick={(c) => { setClaim(c); submit(c); }} />
          </motion.div>
        </div>
        <div className="relative mx-auto grid max-w-6xl gap-3 px-4 pb-20 sm:grid-cols-2 sm:px-6 lg:grid-cols-4">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 + i * 0.06 }}
              className="card p-5"
            >
              <span className="grid h-9 w-9 place-items-center rounded-xl bg-accent-soft text-accent-ink">
                <f.icon className="h-[18px] w-[18px]" />
              </span>
              <h3 className="mt-3 text-[14.5px] font-semibold tracking-tight">{f.title}</h3>
              <p className="mt-1 text-[13px] leading-relaxed text-muted">{f.text}</p>
            </motion.div>
          ))}
        </div>
      </div>
    );
  }

  const result = state.result;
  const explain = state.request?.explain !== false;
  const explanationStreaming = state.status === "running" && state.stages.explanation.status !== "done";
  return (
    <CitationProvider>
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        <div className="mx-auto max-w-4xl lg:max-w-none">{input}</div>
        <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1fr)_300px]">
          <div className="min-w-0 space-y-5">
            {(state.status === "running" || state.totalMs !== null) && (
              <PipelineProgress stages={state.stages} explain={explain} totalMs={state.totalMs} />
            )}
            {state.status === "error" && (
              <div className="card flex items-start gap-3 border-con/30 p-5" role="alert">
                <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-con" />
                <div className="min-w-0 flex-1">
                  <p className="font-semibold">Verification failed</p>
                  <p className="mt-1 text-sm break-words text-muted">{state.error}</p>
                  {source !== "corpus" && (
                    <p className="mt-1 text-sm text-muted">Live sources need internet access; the SciFact corpus works offline.</p>
                  )}
                  <div className="mt-3 flex gap-2">
                    <Button variant="primary" size="sm" onClick={() => submit()}>Try again</Button>
                    {source !== "corpus" && (
                      <Button size="sm" onClick={() => { setSource("corpus"); submit(claim, "corpus"); }}>Use SciFact corpus</Button>
                    )}
                    <Button variant="ghost" size="sm" onClick={reset}>Back</Button>
                  </div>
                </div>
              </div>
            )}
            {!result && state.status === "running" && <CandidateList candidates={state.candidates} running />}
            {result && <VerdictCard result={result} />}
            {result && explain && (state.explanation || explanationStreaming) && (
              <ExplanationCard
                text={state.explanation}
                backend={state.explanationFinal?.backend ?? result.explanation_backend}
                error={state.explanationFinal?.error}
                streaming={explanationStreaming}
                documents={result.documents}
              />
            )}
            {result && <EvidenceSection documents={result.documents} />}
          </div>
          <aside className="space-y-4 lg:sticky lg:top-20 lg:self-start">
            {result ? (
              <SidePanel result={result} onRerun={() => submit(result.claim, result.source === "custom" ? source : result.source)} />
            ) : (
              state.status === "running" && (
                <div className="card p-4 text-[13px] leading-relaxed text-muted">
                  Streaming results as each stage finishes. Retrieval and evidence scoring run locally; nothing leaves this server
                  unless you choose a live source.
                </div>
              )
            )}
          </aside>
        </div>
      </div>
    </CitationProvider>
  );
}
