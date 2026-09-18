# Retrieval evaluation

_Generated 2026-09-18 · 6 questions · top_k=5 (from 20 candidates) · embedding `all-MiniLM-L6-v2` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`_

| Pipeline | hit@5 | MRR |
|----------|-------|-----|
| Dense only (FAISS cosine) | 0.83 | 0.71 |
| Dense + cross-encoder rerank | 1.00 | 0.81 |

**hit@k** = fraction of questions with a relevant passage in the top-k.
**MRR** = mean reciprocal rank of the first relevant passage (higher = it ranks nearer the top).
