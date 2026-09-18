import logging
from typing import Any, List

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = settings.embedding_model,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        model: SentenceTransformer = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model if model is not None else SentenceTransformer(model_name)

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info("Split %d documents into %d chunks.", len(documents), len(chunks))
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        logger.info("Generating embeddings for %d chunks...", len(texts))
        # normalize_embeddings=True -> unit vectors so cosine similarity works with inner product
        return self.model.encode(texts, normalize_embeddings=True)


# Example usage
if __name__ == "__main__":
    from src.data_loader import load_all_documents

    logging.basicConfig(level=logging.INFO)
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
