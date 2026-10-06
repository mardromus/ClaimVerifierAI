# ClaimVerifier AI - evaluation of 'lite' on SciFact dev

## Components

- **retrieval**: `{"embedder": "lsa", "dim": 384, "index": "flat", "hybrid_bm25": true}`
- **rationale**: `{"method": "features", "embedder": {"embedder": "lsa", "dim": 384}}`
- **nli**: `{"method": "lite", "stance_only": true}`
- **decision**: `{"threshold": 0.65, "nei_weight": 0.75, "calibrated": true}`
- **explanation**: `{"backend": "none"}`

Claims evaluated: **300** (Supported: 124, Contradicted: 64, Insufficient Evidence: 112); 43.2 ms/claim.

## Verdict classification (claim level)

| Metric | Value |
|---|---|
| Accuracy | 52.3 |
| Macro Precision | 47.2 |
| Macro Recall | 46.5 |
| Macro F1 | 45.7 |
| Weighted F1 | 50.2 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supported | 54.2 | 67.7 | 60.2 | 124 |
| Contradicted | 32.3 | 15.6 | 21.1 | 64 |
| Insufficient Evidence | 55.3 | 56.2 | 55.8 | 112 |

Confusion matrix (rows = gold, columns = predicted):

| | Supported | Contradicted | Insufficient Evidence |
|---|---|---|---|
| **Supported** | 84 | 13 | 27 |
| **Contradicted** | 30 | 10 | 24 |
| **Insufficient Evidence** | 41 | 8 | 63 |

## Evidence retrieval (188 claims with gold evidence abstracts)

| Metric | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| Recall | 69.0 | 80.1 | 86.6 | 91.6 | 93.8 |
| Hit rate | 70.2 | 81.9 | 88.3 | 93.6 | 95.7 |

**MRR**: 0.7760 (ranking depth 100)

## SciFact abstract-level evaluation

| | Precision | Recall | F1 |
|---|---|---|---|
| Label-only | 39.2 | 18.2 | 24.8 |
| Rationalized | 34.0 | 15.8 | 21.6 |
| Sentence selection | 40.0 | 14.8 | 21.6 |
