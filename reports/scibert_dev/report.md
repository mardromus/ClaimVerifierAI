# ClaimVerifier AI - evaluation of 'scibert' on SciFact dev

## Components

- **retrieval**: `{"embedder": "lsa", "dim": 384, "index": "flat", "hybrid_bm25": true, "rrf_weights": {"dense": 0.0, "bm25": 1.0}}`
- **rationale**: `{"method": "scibert", "model": "models/scibert-rationale"}`
- **nli**: `{"method": "transformer", "model": "models/scibert-verifier"}`
- **decision**: `{"threshold": 0.5, "nei_weight": 1.0, "calibrated": false}`
- **explanation**: `{"backend": "none"}`

Claims evaluated: **300** (Supported: 124, Contradicted: 64, Insufficient Evidence: 112); 2071.4 ms/claim.

## Verdict classification (claim level)

| Metric | Value |
|---|---|
| Accuracy | 66.3 |
| Macro Precision | 63.8 |
| Macro Recall | 64.1 |
| Macro F1 | 63.7 |
| Weighted F1 | 66.4 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supported | 75.0 | 62.9 | 68.4 | 124 |
| Contradicted | 45.7 | 50.0 | 47.8 | 64 |
| Insufficient Evidence | 70.6 | 79.5 | 74.8 | 112 |

Confusion matrix (rows = gold, columns = predicted):

| | Supported | Contradicted | Insufficient Evidence |
|---|---|---|---|
| **Supported** | 78 | 25 | 21 |
| **Contradicted** | 16 | 32 | 16 |
| **Insufficient Evidence** | 10 | 13 | 89 |

## Evidence retrieval (188 claims with gold evidence abstracts)

| Metric | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| Recall | 69.0 | 80.1 | 86.6 | 91.6 | 93.8 |
| Hit rate | 70.2 | 81.9 | 88.3 | 93.6 | 95.7 |

**MRR**: 0.7760 (ranking depth 100)

## SciFact abstract-level evaluation

| | Precision | Recall | F1 |
|---|---|---|---|
| Label-only | 36.8 | 44.0 | 40.1 |
| Rationalized | 33.6 | 40.2 | 36.6 |
| Sentence selection | 42.8 | 51.4 | 46.7 |
