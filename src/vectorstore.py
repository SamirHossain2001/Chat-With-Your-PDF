import json
import logging
import os
from typing import Any, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.embedding import EmbeddingPipeline

logger = logging.getLogger(__name__)


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: Optional[str] = "faiss_store",
        embedding_model: str = settings.embedding_model,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        model: SentenceTransformer = None,
    ):
        # persist_dir=None keeps the index in memory only (used for per-session uploads)
        self.persist_dir = persist_dir
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = model if model is not None else SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_from_documents(self, documents: List[Any]):
        logger.info("Building vector store from %d raw documents...", len(documents))
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            model=self.model,
        )
        chunks = emb_pipe.chunk_documents(documents)
        if not chunks:
            raise ValueError("No text could be extracted from the documents.")
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [
            {
                "text": chunk.page_content,
                "source": os.path.basename(str(chunk.metadata.get("source", ""))),
                "page": chunk.metadata.get("page"),
            }
            for chunk in chunks
        ]
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        if self.persist_dir:
            self.save()

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            # Inner product on normalized vectors == cosine similarity
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        logger.info("Added %d vectors to Faiss index.", embeddings.shape[0])

    def save(self):
        faiss.write_index(self.index, os.path.join(self.persist_dir, "faiss.index"))
        # JSON (not pickle) — safe to load, plain metadata only
        with open(os.path.join(self.persist_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        logger.info("Saved Faiss index and metadata to %s", self.persist_dir)

    def load(self):
        self.index = faiss.read_index(os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "metadata.json"), encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.info("Loaded Faiss index and metadata from %s", self.persist_dir)

    def search(self, query_embedding: np.ndarray, top_k: int = settings.top_k):
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            # FAISS returns -1 when top_k exceeds the number of stored vectors
            if idx < 0:
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": int(idx), "score": float(score), "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = settings.top_k):
        query_emb = self.model.encode([query_text], normalize_embeddings=True).astype("float32")
        return self.search(query_emb, top_k=top_k)


# Example usage
if __name__ == "__main__":
    from src.data_loader import load_all_documents

    logging.basicConfig(level=logging.INFO)
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
