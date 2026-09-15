import hashlib
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

from src.data_loader import load_pdf
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen/qwen3.8-27b"
# Every question costs Groq quota on the owner's API key, so cap it per browser session
MAX_QUESTIONS_PER_SESSION = 30
SUGGESTED_QUESTIONS = [
    "Summarize this document",
    "What are the key points?",
    "What problem does this solve?",
]

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
    h1 { font-size: clamp(1.8rem, 5vw, 2.6rem) !important; }
    .hero-sub { font-size: 1.05rem; opacity: 0.8; margin-top: -0.5rem; margin-bottom: 1.25rem; }
    .step { border: 1px solid rgba(128,128,128,0.25); border-radius: 0.75rem;
            padding: 0.9rem 1rem; height: 100%; }
    .step b { display: block; margin-bottom: 0.2rem; }
    .step span { font-size: 0.9rem; opacity: 0.75; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the embedding model (first visit only)...")
def get_embedding_model():
    # Shared across all sessions: loaded once per server process
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        st.error("This app isn't configured yet: GROQ_API_KEY is missing. Add it to .env or the Streamlit secrets.")
        st.stop()
    return ChatGroq(groq_api_key=api_key, model_name=LLM_MODEL)


def reset_documents():
    for key in ("rag", "file_hashes", "doc_info", "messages"):
        st.session_state.pop(key, None)


def build_rag(uploaded_files):
    documents = []
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        try:
            documents.extend(load_pdf(tmp_path, source_name=uploaded.name))
        finally:
            os.remove(tmp_path)

    if not any(doc.page_content.strip() for doc in documents):
        raise ValueError("no text could be extracted. Scanned or image-only PDFs aren't supported yet.")

    # persist_dir=None: in-memory index, private to this browser session
    store = FaissVectorStore(persist_dir=None, embedding_model=EMBEDDING_MODEL, model=get_embedding_model())
    store.build_from_documents(documents)
    doc_info = {
        "names": [f.name for f in uploaded_files],
        "pages": len(documents),
        "chunks": len(store.metadata),
    }
    return RAGSearch(vectorstore=store, llm=get_llm()), doc_info


def friendly_error(e: Exception) -> str:
    text = str(e).lower()
    if "rate limit" in text or "429" in text:
        return "The service is busy right now (rate limit reached). Please wait a minute and try again."
    if "api key" in text or "401" in text:
        return "The app's API key is invalid. Please let the app owner know."
    return f"Something went wrong while answering: {e}"


def snippet(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon=":material/menu_book:"):
        for s in sources:
            page = s.get("page")
            label = f"**{s.get('source', 'document')}**" + (f" · page {page + 1}" if page is not None else "")
            st.markdown(f"{label}\n\n> {snippet(s.get('text', ''))}")


# ---------- Header ----------
st.title("📄 Chat with your PDF")
st.markdown(
    '<p class="hero-sub">Upload a PDF and ask questions about it. Every answer shows the pages it came from.</p>',
    unsafe_allow_html=True,
)

has_docs = "rag" in st.session_state

# ---------- Upload (main area, so it's visible on phones too) ----------
if has_docs:
    info = st.session_state.doc_info
    upload_label = f"{', '.join(info['names'])} · {info['pages']} pages"
else:
    upload_label = "Upload your PDF"

with st.expander(upload_label, expanded=not has_docs, icon=":material/upload_file:"):
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type="pdf",
        accept_multiple_files=True,
        key="uploads",
        help="Text-based PDFs up to 25 MB each.",
    )
    st.caption("🔒 Your files are only used in this browser tab and are discarded when you close it.")

if uploaded_files:
    hashes = sorted(hashlib.sha256(f.getvalue()).hexdigest() for f in uploaded_files)
    if st.session_state.get("file_hashes") != hashes:
        with st.status("Reading your PDF...", expanded=False) as status:
            try:
                rag, doc_info = build_rag(uploaded_files)
                st.session_state.rag = rag
                st.session_state.doc_info = doc_info
                st.session_state.file_hashes = hashes
                st.session_state.messages = []
                status.update(label="Ready! Ask your first question.", state="complete")
            except Exception as e:
                reset_documents()
                status.update(label="Couldn't read that PDF", state="error")
                st.error(f"Sorry, {e}" if isinstance(e, ValueError) else f"Could not process the PDF: {e}")
                st.stop()
        st.rerun()
elif has_docs:
    reset_documents()
    st.rerun()

st.session_state.setdefault("questions_asked", 0)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("About")
    st.markdown(
        "This app finds the passages in your PDF that are most relevant to your question, "
        "then asks an AI model to answer **using only those passages**."
    )
    if "rag" in st.session_state:
        info = st.session_state.doc_info
        st.divider()
        st.markdown(f"**{len(info['names'])} file(s)** · {info['pages']} pages · {info['chunks']} passages")
        if st.button("New chat", icon=":material/refresh:", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    st.divider()
    remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.questions_asked
    st.caption(f"Questions left this session: {max(remaining, 0)} of {MAX_QUESTIONS_PER_SESSION}")

# ---------- Empty state ----------
if "rag" not in st.session_state:
    cols = st.columns(3)
    steps = [
        ("1. Upload", "Add a text-based PDF: a paper, report, manual or book."),
        ("2. Ask", "Type any question in plain language."),
        ("3. Verify", "Open Sources to see the exact pages used."),
    ]
    for col, (title, body) in zip(cols, steps):
        col.markdown(f'<div class="step"><b>{title}</b><span>{body}</span></div>', unsafe_allow_html=True)
    st.stop()

# ---------- Chat ----------
USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/description:"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))

pending = None
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    picked = st.pills("Suggested questions", SUGGESTED_QUESTIONS, label_visibility="collapsed")
    if picked:
        pending = picked

limit_reached = st.session_state.questions_asked >= MAX_QUESTIONS_PER_SESSION
typed = st.chat_input(
    "Question limit reached for this session" if limit_reached else "Ask a question about your document",
    disabled=limit_reached,
)
query = typed or pending

if query and not limit_reached:
    st.session_state.questions_asked += 1
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)

    rag = st.session_state.rag
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        sources = []
        try:
            with st.spinner("Searching the document..."):
                sources = rag.retrieve(query, top_k=5)
            if sources:
                answer = st.write_stream(rag.stream_answer(query, sources))
            else:
                answer = "I couldn't find anything relevant in the document."
                st.markdown(answer)
        except Exception as e:
            answer = friendly_error(e)
            sources = []
            st.error(answer)
        render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    if pending:
        # Clear the selected suggestion so the pills disappear on the next run
        st.rerun()
