# ClaimVerifier AI - evaluation of 'verisci' on SciFact dev

## Components

- **retrieval**: `{"embedder": "lsa", "dim": 384, "index": "flat", "hybrid_bm25": true, "rrf_weights": {"dense": 0.0, "bm25": 1.0}}`
- **rationale**: `{"method": "features", "embedder": {"embedder": "lsa", "dim": 384}}`
- **nli**: `{"method": "transformer", "model": "models/verisci-roberta-large"}`
- **decision**: `{"threshold": 0.5, "nei_weight": 1.0, "calibrated": false}`
- **explanation**: `{"backend": "none"}`

Claims evaluated: **300** (Supported: 124, Contradicted: 64, Insufficient Evidence: 112); 1477.5 ms/claim.

## Verdict classification (claim level)

| Metric | Value |
|---|---|
| Accuracy | 60.0 |
| Macro Precision | 61.6 |
| Macro Recall | 58.6 |
| Macro F1 | 58.7 |
| Weighted F1 | 59.9 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supported | 75.9 | 53.2 | 62.6 | 124 |
| Contradicted | 56.4 | 48.4 | 52.1 | 64 |
| Insufficient Evidence | 52.5 | 74.1 | 61.5 | 112 |

Confusion matrix (rows = gold, columns = predicted):

| | Supported | Contradicted | Insufficient Evidence |
|---|---|---|---|
| **Supported** | 66 | 10 | 48 |
| **Contradicted** | 6 | 31 | 27 |
| **Insufficient Evidence** | 15 | 14 | 83 |

## Evidence retrieval (188 claims with gold evidence abstracts)

| Metric | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| Recall | 69.0 | 80.1 | 86.6 | 91.6 | 93.8 |
| Hit rate | 70.2 | 81.9 | 88.3 | 93.6 | 95.7 |

**MRR**: 0.7760 (ranking depth 100)

## SciFact abstract-level evaluation

| | Precision | Recall | F1 |
|---|---|---|---|
| Label-only | 44.6 | 27.8 | 34.2 |
| Rationalized | 42.3 | 26.3 | 32.4 |
| Sentence selection | 40.3 | 19.9 | 26.7 |
