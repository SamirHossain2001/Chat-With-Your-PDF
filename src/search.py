import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

ANSWER_PROMPT = """You are a helpful assistant answering questions about the user's uploaded documents.
Answer the question using ONLY the context below. If the answer is not in the context,
reply exactly: "I couldn't find that in the document."
Write a clear, well-structured answer in Markdown (short paragraphs, bullet points where useful).

Context:
{context}

Question: {query}

Answer:"""

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "qwen/qwen3.8-27b", vectorstore: FaissVectorStore = None, llm=None):
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
            # Load or build vectorstore
            faiss_path = os.path.join(persist_dir, "faiss.index")
            meta_path = os.path.join(persist_dir, "metadata.pkl")
            if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
                from src.data_loader import load_all_documents
                docs = load_all_documents("data")
                self.vectorstore.build_from_documents(docs)
            else:
                self.vectorstore.load()
        if llm is not None:
            self.llm = llm
        else:
            groq_api_key = os.getenv("GROQ_API_KEY")
            self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
            print(f"[INFO] Groq LLM initialized: {llm_model}")

    def retrieve(self, query: str, top_k: int = 5) -> list:
        results = self.vectorstore.query(query, top_k=top_k)
        return [r["metadata"] for r in results if r["metadata"]]

    def _build_prompt(self, query: str, sources: list) -> str:
        blocks = []
        for s in sources:
            page = s.get("page")
            label = f"[{s.get('source', 'document')}" + (f", page {page + 1}]" if page is not None else "]")
            blocks.append(f"{label}\n{s.get('text', '')}")
        return ANSWER_PROMPT.format(context="\n\n".join(blocks), query=query)

    def stream_answer(self, query: str, sources: list):
        """Yield the answer text piece by piece as the LLM generates it."""
        for chunk in self.llm.stream([self._build_prompt(query, sources)]):
            if chunk.content:
                yield chunk.content

    def answer(self, query: str, top_k: int = 5) -> dict:
        sources = self.retrieve(query, top_k=top_k)
        if not sources:
            return {"answer": "No relevant documents found.", "sources": []}
        response = self.llm.invoke([self._build_prompt(query, sources)])
        return {"answer": response.content, "sources": sources}

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
