export type Verdict = "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE";
export type SourceName = "corpus" | "europepmc" | "pubmed" | "custom";
export type StageName = "retrieval" | "rationale" | "nli" | "explanation";

export interface DocMeta {
  source?: string;
  journal?: string;
  year?: string;
  authors?: string;
  pmid?: string;
  doi?: string;
  url?: string;
}

export interface EvidenceSentence {
  index: number;
  text: string;
  rationale_score: number;
  stance: Verdict;
  stance_probs: Record<Verdict, number>;
}

export interface DocumentResult {
  doc_id: number;
  title: string;
  url: string;
  citation: number;
  rank: number;
  retrieval_rank: number;
  retrieval_score: number;
  dense_score: number;
  bm25_score: number;
  relevance: number;
  has_evidence: boolean;
  evidence_weight: number;
  stance: Verdict;
  stance_probs: Record<Verdict, number>;
  evidence: EvidenceSentence[];
  sentences: string[];
  sentence_scores: number[];
  meta: DocMeta;
}

export interface VerificationResult {
  claim: string;
  verdict: Verdict;
  verdict_display: string;
  confidence: number;
  scores: Record<Verdict, number>;
  support_strength: number;
  contradict_strength: number;
  mixed_evidence: boolean;
  documents: DocumentResult[];
  explanation: string;
  explanation_backend: string;
  citations: number[];
  timings_ms: Record<string, number>;
  components: Record<string, Record<string, unknown>>;
  source: SourceName;
  candidates_scanned: number;
}

export interface CandidatePreview {
  doc_id: number;
  title: string;
  url: string;
  rank: number;
  dense_score: number;
  bm25_score: number;
  meta: DocMeta;
  num_sentences: number;
}

export interface StageEvent {
  stage: StageName;
  status: "start" | "done";
  ms?: number;
  count?: number;
  source?: string;
  sentences?: number;
  kept?: number;
  backend?: string;
}

export interface ExplanationEvent {
  text: string;
  backend: string;
  citations: number[];
  error?: string | null;
}

export interface SourceInfo {
  name: SourceName;
  label: string;
  description: string;
}

export interface ApiInfo {
  version: string;
  name: string;
  components: Record<string, Record<string, unknown>>;
  stats: { documents: number; train_claims?: number; dev_claims?: number };
  sources: SourceInfo[];
  labels: Record<Verdict, string>;
  defaults: { top_k: number; rerank_depth: number; explanation_backend: string; explanation_model: string };
}

export interface ExampleClaim {
  id: number;
  claim: string;
  label: Verdict;
  label_display: string;
}

export interface CustomDocument {
  title: string;
  abstract: string;
}

export interface VerifyRequest {
  claim: string;
  top_k?: number;
  explain?: boolean;
  source?: SourceName;
  documents?: CustomDocument[];
}

export interface ReportSummary {
  name: string;
  num_claims: number;
  accuracy: number;
  macro_f1: number;
  macro_precision: number;
  macro_recall: number;
  recall_at_5: number | null;
  mrr: number | null;
  abstract_label_f1: number;
  components: Record<string, Record<string, unknown>>;
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ReportMetrics {
  num_claims: number;
  label_distribution: Record<Verdict, number>;
  verdict_classification: {
    accuracy: number;
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
    weighted_f1: number;
    per_class: Record<Verdict, ClassMetrics>;
    confusion_matrix: { labels: Verdict[]; matrix: number[][] };
  };
  evidence_retrieval: Record<string, number>;
  scifact_abstract_level: {
    label_only: { precision: number; recall: number; f1: number };
    rationalized: { precision: number; recall: number; f1: number };
  };
  scifact_sentence_selection: { precision: number; recall: number; f1: number };
  ms_per_claim: number;
  components: Record<string, Record<string, unknown>>;
}
