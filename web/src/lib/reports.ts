const NAMES: Record<string, string> = {
  lite_dev: "Lite (offline)",
  "scibert-rationale+lite-nli_dev": "SciBERT evidence + lite stance",
  scibert_dev: "SciBERT pipeline",
  "scibert-rerank_dev": "SciBERT + re-ranking",
  default_dev: "Default (SBERT · SciBERT · DeBERTa-v3)",
  large_dev: "Large (GPU)",
};

export const reportLabel = (name: string) => NAMES[name] ?? name.replace(/_dev$/, "").replace(/[-_]/g, " ");

/** Series colours follow the report (stable order), never its rank. */
export const SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];

export const MAJORITY = { accuracy: 124 / 300, macro_f1: (2 * (124 / 300)) / (1 + 124 / 300) / 3 };
