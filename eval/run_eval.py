"""Retrieval evaluation: measures hit@k and MRR over a small gold set, and
quantifies how much the cross-encoder reranker improves results vs dense-only.

Run from the project root:  python -m eval.run_eval
Builds an in-memory index from data/pdf, so it needs no Groq key.
"""
import json
import logging
import statistics
from datetime import date
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import settings
from src.data_loader import load_all_documents
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore

GOLD = Path(__file__).parent / "gold.json"
REPORT = Path(__file__).parent / "EVAL.md"


def _is_relevant(text: str, keywords: list) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def _score(rag: RAGSearch, gold: list, use_reranker: bool) -> tuple[float, float]:
    hits, reciprocal_ranks = [], []
    for item in gold:
        results = rag.vectorstore.query(item["question"], top_k=settings.retrieve_candidates)
        metas = [r["metadata"] for r in results if r["metadata"]]
        metas = rag._rerank(item["question"], metas, settings.top_k) if use_reranker else metas[: settings.top_k]
        rank = next((i + 1 for i, m in enumerate(metas) if _is_relevant(m.get("text", ""), item["keywords"])), 0)
        hits.append(1 if rank else 0)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return sum(hits) / len(hits), statistics.mean(reciprocal_ranks)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    gold = json.loads(GOLD.read_text(encoding="utf-8"))

    model = SentenceTransformer(settings.embedding_model)
    store = FaissVectorStore(persist_dir=None, embedding_model=settings.embedding_model, model=model)
    store.build_from_documents(load_all_documents("data/pdf"))
    rag = RAGSearch(vectorstore=store, llm=object(), reranker=CrossEncoder(settings.reranker_model))

    dense_hit, dense_mrr = _score(rag, gold, use_reranker=False)
    rerank_hit, rerank_mrr = _score(rag, gold, use_reranker=True)

    report = f"""# Retrieval evaluation

_Generated {date.today()} · {len(gold)} questions · top_k={settings.top_k} \
(from {settings.retrieve_candidates} candidates) · embedding `{settings.embedding_model}` · \
reranker `{settings.reranker_model}`_

| Pipeline | hit@{settings.top_k} | MRR |
|----------|-------|-----|
| Dense only (FAISS cosine) | {dense_hit:.2f} | {dense_mrr:.2f} |
| Dense + cross-encoder rerank | {rerank_hit:.2f} | {rerank_mrr:.2f} |

**hit@k** = fraction of questions with a relevant passage in the top-k.
**MRR** = mean reciprocal rank of the first relevant passage (higher = it ranks nearer the top).
"""
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
