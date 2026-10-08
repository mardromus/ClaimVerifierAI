import { ArrowRight, BookOpenCheck, Code2, Database, Gauge, Layers, Scale, ScanText, Search, ShieldAlert, Sparkles } from "lucide-react";
import { motion } from "motion/react";
import { useApp } from "../hooks/useApp";
import { compact } from "../lib/format";

const STAGES = [
  { icon: Search, title: "Retrieve", model: "Sentence-BERT + FAISS, fused with BM25", text: "The claim is embedded and matched against every abstract; dense and keyword rankings are merged with reciprocal-rank fusion. Live mode queries Europe PMC or PubMed instead." },
  { icon: ScanText, title: "Select evidence", model: "SciBERT cross-encoder", text: "Every sentence of every candidate abstract is scored as evidence or not. The strongest sentences become the rationale; candidates can be re-ranked by them." },
  { icon: Scale, title: "Infer stance", model: "DeBERTa-v3 NLI (or SciBERT verifier)", text: "The selected evidence is the premise and the claim the hypothesis: entailment → supports, contradiction → contradicts, neutral → insufficient." },
  { icon: Gauge, title: "Aggregate", model: "Relevance-gated evidence fusion", text: "Papers are weighted by how relevant their evidence is; the strongest support and contradiction combine into three probabilities that sum to one." },
  { icon: Sparkles, title: "Explain", model: "Qwen 2.5 / Llama 3 Instruct (RAG)", text: "An LLM explains the verdict using only the numbered evidence, citing it as [n]. Invalid citations are removed; a template is used if no LLM is available." },
];

function Section({ icon: Icon, title, children }: { icon: typeof Search; title: string; children: React.ReactNode }) {
  return (
    <section className="card p-6">
      <h2 className="mb-3 flex items-center gap-2 text-[16px] font-semibold tracking-tight">
        <Icon className="h-4.5 w-4.5 text-accent" /> {title}
      </h2>
      <div className="space-y-3 text-[14px] leading-relaxed text-ink-2">{children}</div>
    </section>
  );
}

export default function AboutPage() {
  const { info } = useApp();
  const comps = info?.components ?? {};
  return (
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <header className="mb-10 max-w-3xl">
        <p className="flex items-center gap-2 text-sm font-medium text-accent-ink">
          <Layers className="h-4 w-4" /> How it works
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">Evidence first, then a verdict, then words</h1>
        <p className="mt-3 text-[15px] leading-relaxed text-muted">
          A search engine lists related papers. ClaimVerifier reads them: every verdict is traced to specific sentences in specific papers, with
          the model's confidence shown at each step. The language model only explains a decision that the NLI pipeline has already made.
        </p>
      </header>

      <ol className="grid gap-3 md:grid-cols-5">
        {STAGES.map((s, i) => (
          <motion.li
            key={s.title}
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.07 }}
            className="card relative p-4"
          >
            <div className="flex items-center gap-2">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-accent-soft text-accent-ink">
                <s.icon className="h-4 w-4" />
              </span>
              <span className="tnum text-xs font-semibold text-muted">0{i + 1}</span>
            </div>
            <h3 className="mt-3 text-[14.5px] font-semibold">{s.title}</h3>
            <p className="mt-0.5 text-[12px] font-medium text-accent-ink">{s.model}</p>
            <p className="mt-2 text-[12.5px] leading-relaxed text-muted">{s.text}</p>
            {i < STAGES.length - 1 && (
              <ArrowRight className="absolute top-1/2 -right-3 z-10 hidden h-4 w-4 -translate-y-1/2 rounded-full bg-bg text-muted md:block" />
            )}
          </motion.li>
        ))}
      </ol>

      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        <Section icon={Database} title="Data">
          <p>
            <b className="font-semibold text-ink">SciFact</b> (Wadden et al., EMNLP 2020) pairs 1,409 expert-written claims with{" "}
            {info ? compact(info.stats.documents) : "5,183"} research abstracts annotated with SUPPORT / CONTRADICT labels and the exact rationale
            sentences. Models are trained on the train split ({info?.stats.train_claims ?? 809} claims) and evaluated on the dev split (
            {info?.stats.dev_claims ?? 300} claims). Live mode searches 40M+ abstracts via Europe PMC or PubMed.
          </p>
        </Section>
        <Section icon={Gauge} title="Turning evidence into a verdict">
          <p>
            For each paper, <i>g</i> = min(1, <i>r</i> / τ) weights its NLI probabilities by evidence relevance <i>r</i>. The strongest weighted
            support <i>S</i> and contradiction <i>C</i> act as two detectors:
          </p>
          <p className="rounded-xl bg-surface-2 px-4 py-3 font-mono text-[12.5px] leading-6">
            P(supported) ∝ S(1−C) + share of SC<br />
            P(contradicted) ∝ C(1−S) + share of SC<br />
            P(insufficient) ∝ w·(1−S)(1−C)
          </p>
          <p className="text-[13px] text-muted">τ and w can be calibrated on the train split. Strong evidence both ways is flagged as mixed.</p>
        </Section>
        <Section icon={Layers} title="Loaded on this server">
          <dl className="grid grid-cols-[110px_1fr] gap-x-3 gap-y-1.5 text-[13px]">
            {Object.entries(comps).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-muted capitalize">{k}</dt>
                <dd className="font-mono text-[12px] break-all text-ink-2">
                  {Object.entries(v)
                    .map(([kk, vv]) => `${kk}=${typeof vv === "object" ? JSON.stringify(vv) : String(vv)}`)
                    .join(" · ")}
                </dd>
              </div>
            ))}
            {!info && <dd className="col-span-2 text-muted">Connecting to the API…</dd>}
          </dl>
        </Section>
        <Section icon={BookOpenCheck} title="Metrics">
          <ul className="list-disc space-y-1 pl-5">
            <li><b className="font-medium text-ink">Accuracy, precision, recall, F1</b> of the three-way claim verdict (macro-averaged).</li>
            <li><b className="font-medium text-ink">Recall@K</b>: share of gold evidence papers among the top K retrieved; <b className="font-medium text-ink">MRR</b>: mean of 1 / rank of the first gold paper.</li>
            <li><b className="font-medium text-ink">SciFact abstract-level F1</b>: per-paper labels, optionally requiring a full gold rationale.</li>
          </ul>
        </Section>
        <Section icon={Code2} title="Use the API">
          <pre className="scrollbar-thin overflow-x-auto rounded-xl bg-surface-2 p-4 font-mono text-[12px] leading-5 text-ink-2">{`curl -N -X POST ${typeof window !== "undefined" ? window.location.origin : ""}/api/verify/stream \\
  -H 'Content-Type: application/json' \\
  -d '{"claim": "Statins reduce cardiovascular events.",
       "source": "corpus", "top_k": 5}'`}</pre>
          <p className="text-[13px] text-muted">
            Streams pipeline stages, the verdict and explanation tokens as server-sent events. Interactive docs at{" "}
            <a className="font-medium text-accent-ink hover:underline" href="/docs">/docs</a>.
          </p>
        </Section>
        <Section icon={ShieldAlert} title="Limitations">
          <ul className="list-disc space-y-1 pl-5">
            <li>“Insufficient evidence” means the retrieved papers don't settle the claim — not that it is false.</li>
            <li>The models judge entailment, not study quality, sample size or publication bias.</li>
            <li>Contradictions are the hardest class; always read the highlighted sentences and the papers.</li>
          </ul>
        </Section>
      </div>
    </div>
  );
}
