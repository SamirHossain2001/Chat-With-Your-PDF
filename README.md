# 📄 Chat with your PDF

Upload a PDF, ask questions, and get answers grounded in your document — with the source pages cited. A simple RAG app: local embeddings (SentenceTransformers) + FAISS search + a Groq-hosted LLM.

**▶️ Live demo: [very-simple-rag.streamlit.app](https://very-simple-rag.streamlit.app/)**

## Run locally

Requires Python 3.12+ and a free [Groq API key](https://console.groq.com).

```bash
git clone https://github.com/SamirHossain2001/Traditional-RAG.git
cd Traditional-RAG

pip install -r requirements.txt      # or: uv sync

echo "GROQ_API_KEY=your_key_here" > .env

streamlit run app.py
```

## How it works

```
PDF ─▶ load pages ─▶ chunk (1000 / 200 overlap) ─▶ embed ─▶ FAISS index ─▶ retrieve top-k ─▶ Groq LLM ─▶ streamed, cited answer
```

- `src/data_loader.py` — load PDF pages
- `src/embedding.py` — chunk + embed
- `src/vectorstore.py` — FAISS index
- `src/search.py` — retrieve + stream the answer
- `app.py` — Streamlit UI

## License

[MIT](LICENSE) © 2026 SamirHossain2001
