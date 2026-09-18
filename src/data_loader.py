import logging
from pathlib import Path
from typing import Any, List

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

logger = logging.getLogger(__name__)

# Glob pattern -> loader class for load_all_documents()
LOADERS = {
    "**/*.pdf": PyPDFLoader,
    "**/*.txt": TextLoader,
    "**/*.csv": CSVLoader,
    "**/*.xlsx": UnstructuredExcelLoader,
    "**/*.docx": Docx2txtLoader,
    "**/*.json": JSONLoader,
}


def load_pdf(path: str, source_name: str = None) -> List[Any]:
    """Load a single PDF into LangChain documents (one per page).

    source_name overrides metadata["source"], e.g. the original upload filename.
    """
    documents = PyPDFLoader(str(path)).load()
    if source_name:
        for doc in documents:
            doc.metadata["source"] = source_name
    logger.debug("Loaded %d PDF pages from %s", len(documents), source_name or path)
    return documents


def load_all_documents(data_dir: str) -> List[Any]:
    """Load every supported file (PDF, TXT, CSV, Excel, Word, JSON) from a directory."""
    data_path = Path(data_dir).resolve()
    documents: List[Any] = []
    for pattern, loader_cls in LOADERS.items():
        for file in data_path.glob(pattern):
            try:
                loaded = loader_cls(str(file)).load()
                documents.extend(loaded)
                logger.debug("Loaded %d docs from %s", len(loaded), file)
            except Exception as e:
                logger.error("Failed to load %s: %s", file, e)
    logger.info("Loaded %d documents from %s", len(documents), data_path)
    return documents


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)
