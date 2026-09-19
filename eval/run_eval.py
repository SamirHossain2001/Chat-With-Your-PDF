"""Evaluation for the RAG pipeline.

Retrieval metrics (hit@k, MRR) always run and need no Groq key — they compare
dense-only vs dense+rerank. Pass --faithfulness to also grade answer groundedness
with an LLM judge (uses GROQ_API_KEY, a few calls per question).

Run from the project root:
    python -m eval.run_eval
    python -m eval.run_eval --faithfulness
"""
import argparse
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

JUDGE_PROMPT = """You are a strict evaluator. Reply with a single word: YES or NO.
Answer YES only if every claim in the ANSWER is supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Supported (YES/NO):"""


def _is_relevant(text: str, keywords: list) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def _retrieval_scores(rag: RAGSearch, gold: list, use_reranker: bool) -> tuple[float, float]:
    hits, reciprocal_ranks = [], []
    for item in gold:
        results = rag.vectorstore.query(item["question"], top_k=settings.retrieve_candidates)
        metas = [r["metadata"] for r in results if r["metadata"]]
        if use_reranker:
            metas = [m for m, _ in rag._rerank(item["question"], metas)[: settings.top_k]]
        else:
            metas = metas[: settings.top_k]
        rank = next((i + 1 for i, m in enumerate(metas) if _is_relevant(m.get("text", ""), item["keywords"])), 0)
        hits.append(1 if rank else 0)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return sum(hits) / len(hits), statistics.mean(reciprocal_ranks)


def _faithfulness(rag: RAGSearch, gold: list) -> float:
    supported = 0
    for item in gold:
        result = rag.answer(item["question"])
        context = "\n\n".join(s.get("text", "") for s in result["sources"])
        verdict = rag._invoke(JUDGE_PROMPT.format(context=context, answer=result["answer"]))
        if verdict.strip().lower().startswith("yes"):
            supported += 1
    return supported / len(gold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faithfulness", action="store_true", help="Grade answer groundedness (needs GROQ_API_KEY).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    gold = json.loads(GOLD.read_text(encoding="utf-8"))

    model = SentenceTransformer(settings.embedding_model)
    store = FaissVectorStore(persist_dir=None, embedding_model=settings.embedding_model, model=model)
    store.build_from_documents(load_all_documents("data/pdf"))

    llm = object()  # placeholder; retrieval metrics never call the LLM
    if args.faithfulness:
        from langchain_groq import ChatGroq

        llm = ChatGroq(groq_api_key=settings.groq_api_key, model_name=settings.llm_model)
    rag = RAGSearch(vectorstore=store, llm=llm, reranker=CrossEncoder(settings.reranker_model))

    dense_hit, dense_mrr = _retrieval_scores(rag, gold, use_reranker=False)
    rerank_hit, rerank_mrr = _retrieval_scores(rag, gold, use_reranker=True)

    faith_line = ""
    if args.faithfulness:
        faith = _faithfulness(rag, gold)
        faith_line = f"\n**Answer faithfulness (reranked pipeline):** {faith:.0%} of answers fully supported by the retrieved context.\n"

    report = f"""# Retrieval evaluation

_Generated {date.today()} · {len(gold)} questions · top_k={settings.top_k} \
(from {settings.retrieve_candidates} candidates) · embedding `{settings.embedding_model}` · \
reranker `{settings.reranker_model}`_

| Pipeline | hit@{settings.top_k} | MRR |
|----------|-------|-----|
| Dense only (FAISS cosine) | {dense_hit:.2f} | {dense_mrr:.2f} |
| Dense + cross-encoder rerank | {rerank_hit:.2f} | {rerank_mrr:.2f} |
{faith_line}
**hit@k** = fraction of questions with a relevant passage in the top-k.
**MRR** = mean reciprocal rank of the first relevant passage (higher = it ranks nearer the top).
"""
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
