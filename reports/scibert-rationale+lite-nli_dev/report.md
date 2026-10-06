# ClaimVerifier AI - evaluation of 'scibert-rationale+lite-nli' on SciFact dev

## Components

- **retrieval**: `{"embedder": "lsa", "dim": 384, "index": "flat", "hybrid_bm25": true, "rrf_weights": {"dense": 0.0, "bm25": 1.0}}`
- **rationale**: `{"method": "scibert", "model": "models/scibert-rationale"}`
- **nli**: `{"method": "lite", "stance_only": true}`
- **decision**: `{"threshold": 0.5, "nei_weight": 1.0, "calibrated": false}`
- **explanation**: `{"backend": "none"}`

Claims evaluated: **300** (Supported: 124, Contradicted: 64, Insufficient Evidence: 112); 5443.2 ms/claim.

## Verdict classification (claim level)

| Metric | Value |
|---|---|
| Accuracy | 63.3 |
| Macro Precision | 60.6 |
| Macro Recall | 57.7 |
| Macro F1 | 57.9 |
| Weighted F1 | 62.1 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supported | 59.8 | 79.0 | 68.1 | 124 |
| Contradicted | 45.0 | 28.1 | 34.6 | 64 |
| Insufficient Evidence | 77.1 | 66.1 | 71.2 | 112 |

Confusion matrix (rows = gold, columns = predicted):

| | Supported | Contradicted | Insufficient Evidence |
|---|---|---|---|
| **Supported** | 98 | 14 | 12 |
| **Contradicted** | 36 | 18 | 10 |
| **Insufficient Evidence** | 30 | 8 | 74 |

## Evidence retrieval (188 claims with gold evidence abstracts)

| Metric | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| Recall | 69.0 | 80.1 | 86.6 | 91.6 | 93.8 |
| Hit rate | 70.2 | 81.9 | 88.3 | 93.6 | 95.7 |

**MRR**: 0.7760 (ranking depth 100)

## SciFact abstract-level evaluation

| | Precision | Recall | F1 |
|---|---|---|---|
| Label-only | 31.9 | 49.8 | 38.9 |
| Rationalized | 29.1 | 45.5 | 35.5 |
| Sentence selection | 37.2 | 56.3 | 44.8 |
