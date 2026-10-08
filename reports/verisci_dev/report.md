# ClaimVerifier AI - evaluation of 'verisci' on SciFact dev

## Components

- **retrieval**: `{"embedder": "lsa", "dim": 384, "index": "flat", "hybrid_bm25": true, "rrf_weights": {"dense": 0.0, "bm25": 1.0}}`
- **rationale**: `{"method": "scibert", "model": "models/scibert-rationale"}`
- **nli**: `{"method": "transformer", "model": "models/verisci-roberta-large"}`
- **decision**: `{"threshold": 0.5, "nei_weight": 1.0, "calibrated": false}`
- **explanation**: `{"backend": "none"}`

Claims evaluated: **300** (Supported: 124, Contradicted: 64, Insufficient Evidence: 112); 2952.3 ms/claim.

## Verdict classification (claim level)

| Metric | Value |
|---|---|
| Accuracy | 69.7 |
| Macro Precision | 69.3 |
| Macro Recall | 67.0 |
| Macro F1 | 67.7 |
| Weighted F1 | 69.4 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supported | 75.0 | 72.6 | 73.8 | 124 |
| Contradicted | 67.3 | 51.6 | 58.4 | 64 |
| Insufficient Evidence | 65.6 | 76.8 | 70.8 | 112 |

Confusion matrix (rows = gold, columns = predicted):

| | Supported | Contradicted | Insufficient Evidence |
|---|---|---|---|
| **Supported** | 90 | 8 | 26 |
| **Contradicted** | 12 | 33 | 19 |
| **Insufficient Evidence** | 18 | 8 | 86 |

## Evidence retrieval (188 claims with gold evidence abstracts)

| Metric | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| Recall | 69.0 | 80.1 | 86.6 | 91.6 | 93.8 |
| Hit rate | 70.2 | 81.9 | 88.3 | 93.6 | 95.7 |

**MRR**: 0.7760 (ranking depth 100)

## SciFact abstract-level evaluation

| | Precision | Recall | F1 |
|---|---|---|---|
| Label-only | 49.4 | 56.0 | 52.5 |
| Rationalized | 46.0 | 52.2 | 48.9 |
| Sentence selection | 47.4 | 51.6 | 49.4 |
