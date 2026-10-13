# ClaimVerifier AI - evaluation of 'default' on SciFact dev

## Components

- **retrieval**: `{"embedder": "sbert", "model_name": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384, "index": "flat", "hybrid_bm25": true, "rrf_weights": {"dense": 1.0, "bm25": 1.0}}`
- **rationale**: `{"method": "scibert", "model": "models/scibert-rationale"}`
- **nli**: `{"method": "transformer", "model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"}`
- **decision**: `{"threshold": 0.5, "nei_weight": 1.0, "calibrated": false}`
- **explanation**: `{"backend": "none"}`

Claims evaluated: **300** (Supported: 124, Contradicted: 64, Insufficient Evidence: 112); 10354.2 ms/claim.

## Verdict classification (claim level)

| Metric | Value |
|---|---|
| Accuracy | 65.3 |
| Macro Precision | 69.9 |
| Macro Recall | 65.5 |
| Macro F1 | 64.7 |
| Weighted F1 | 65.1 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supported | 91.2 | 50.0 | 64.6 | 124 |
| Contradicted | 62.5 | 62.5 | 62.5 | 64 |
| Insufficient Evidence | 56.0 | 83.9 | 67.1 | 112 |

Confusion matrix (rows = gold, columns = predicted):

| | Supported | Contradicted | Insufficient Evidence |
|---|---|---|---|
| **Supported** | 62 | 10 | 52 |
| **Contradicted** | 2 | 40 | 22 |
| **Insufficient Evidence** | 4 | 14 | 94 |

## Evidence retrieval (188 claims with gold evidence abstracts)

| Metric | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| Recall | 66.8 | 86.0 | 89.4 | 94.3 | 95.8 |
| Hit rate | 68.6 | 88.8 | 91.5 | 95.7 | 96.8 |

**MRR**: 0.7898 (ranking depth 100)

## SciFact abstract-level evaluation

| | Precision | Recall | F1 |
|---|---|---|---|
| Label-only | 58.3 | 42.1 | 48.9 |
| Rationalized | 57.0 | 41.1 | 47.8 |
| Sentence selection | 52.1 | 40.7 | 45.7 |
