# ui/streamlit_app.py
# SmartDocs — AI-Powered Business Assistant
# Full RAG Integration — Final Version

import streamlit as st
import sys
import os

# ── Path fix ──────────────────────────────────────────────────────────────────
# Ensures Python can find the app/ modules from the ui/ subfolder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion import load_documents, split_documents
from app.embeddings import create_vectorstore, load_vectorstore, vectorstore_exists
from app.retriever import get_retriever
from app.chain import build_qa_chain

# ── Page Configuration ────────────────────────────────────────────────────────
# Must be the FIRST Streamlit command — nothing else before this
st.set_page_config(
    page_title="SmartDocs Assistant",
    page_icon="🤖",
    layout="wide"
)

# ── Session State Initialization ──────────────────────────────────────────────
# Only sets defaults on first load — preserved across reruns after that
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # List of {question, answer, sources}

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None            # Built RAG chain

if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False     # True once docs are processed

if "processing_error" not in st.session_state:
    st.session_state.processing_error = None    # Stores error message if any

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0           # How many chunks were indexed

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []   # Names of processed files


# ── Helper Functions ──────────────────────────────────────────────────────────

def save_uploaded_file(uploaded_file):
    """
    Streamlit's uploaded file is an in-memory object.
    LangChain loaders need a real file path on disk.
    This saves it to data/uploads/ and returns the path.
    """
    save_dir = "data/uploads"
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def process_documents(uploaded_files):
    """
    Orchestrates the full RAG pipeline:
    Save → Load → Chunk → Embed → VectorStore → Retriever → Chain

    Returns: (qa_chain, total_chunks, file_names)
    """
    all_chunks = []
    file_names = []

    for uploaded_file in uploaded_files:

        # Guard: skip empty files
        if uploaded_file.size == 0:
            st.warning(f"⚠️ Skipping empty file: {uploaded_file.name}")
            continue

        file_path = save_uploaded_file(uploaded_file)
        docs = load_documents(file_path)
        chunks = split_documents(docs)
        all_chunks.extend(chunks)
        file_names.append(uploaded_file.name)

    # Guard: nothing extracted
    if not all_chunks:
        raise ValueError(
            "No content could be extracted from the uploaded files. "
            "Please check your files and try again."
        )

    # Build the full RAG stack
    vectorstore = create_vectorstore(all_chunks)
    retriever = get_retriever(vectorstore, k=3)
    qa_chain = build_qa_chain(retriever)

    return qa_chain, len(all_chunks), file_names


def ask_question(question):
    """
    Sends a question through the RAG chain stored in session state.
    Returns (answer, source_documents).
    """
    result = st.session_state.qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]
    return answer, sources


def render_source_expander(sources):
    """
    Renders the collapsible source chunk expander.
    Extracted into its own function since we use it in two places:
    1. When rendering chat history
    2. When rendering the live new answer
    """
    if sources:
        with st.expander("📚 View Source Chunks"):
            for i, source in enumerate(sources):
                st.caption(f"**Source Chunk {i + 1}:**")
                st.caption(source.page_content[:300] + "...")
                if i < len(sources) - 1:
                    st.divider()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# Admin panel — business owner uploads documents here

with st.sidebar:
    st.title("🤖 SmartDocs")
    st.caption("AI-Powered Business Assistant")
    st.divider()

    # ── Upload section ──
    st.subheader("📁 Upload Documents")
    st.write(
        "Upload your business PDFs or text files. "
        "The assistant will answer questions using only these documents."
    )

    uploaded_files = st.file_uploader(
        label="Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT. Max 200MB per file."
    )

    # Show selected file names
    if uploaded_files:
        for f in uploaded_files:
            st.caption(f"📄 {f.name} ({round(f.size / 1024, 1)} KB)")

    # Process button — only show when files are selected
    if uploaded_files:
        st.divider()
        if st.button(
            label="⚡ Process Documents",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Reading and indexing your documents..."):
                try:
                    qa_chain, total_chunks, file_names = process_documents(uploaded_files)

                    # Store everything in session state
                    st.session_state.qa_chain = qa_chain
                    st.session_state.docs_processed = True
                    st.session_state.total_chunks = total_chunks
                    st.session_state.uploaded_file_names = file_names
                    st.session_state.chat_history = []      # Fresh chat for new docs
                    st.session_state.processing_error = None

                    st.success(f"✅ Indexed {total_chunks} chunks from {len(file_names)} file(s).")

                except Exception as e:
                    st.session_state.processing_error = str(e)
                    st.session_state.docs_processed = False
                    st.error(f"❌ Error: {str(e)}")

    # Show processing error persistently
    if st.session_state.processing_error:
        st.error(f"❌ Last error: {st.session_state.processing_error}")

    st.divider()

    # ── Status section ──
    st.subheader("📊 Status")

    if st.session_state.docs_processed:
        st.success("🟢 Assistant is active")
        st.caption(f"📦 Chunks indexed: {st.session_state.total_chunks}")
        st.caption(f"💬 Messages: {len(st.session_state.chat_history)}")

    else:
        st.warning("🟡 Awaiting documents")

    # ── Clear chat button ──
    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
# Customer-facing chat interface

st.title("💬 Ask Your Business Assistant")
st.caption(
    "Answers are generated strictly from your uploaded documents. "
    "No information is fabricated or assumed."
)
st.divider()

# ── Empty / Welcome state ──
if not st.session_state.chat_history:
    if not st.session_state.docs_processed:
        st.info(
            "👈 **Get started:** Upload your business documents "
            "in the sidebar and click **⚡ Process Documents**."
        )
    else:
        st.info("✅ Documents loaded! Type your first question below.")

# ── Render chat history ──
# Loop through all past Q&A and render as chat bubbles
for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])
        render_source_expander(chat["sources"])

# ── Chat input ──
# Pinned to bottom of screen automatically by Streamlit
# Disabled until documents are processed
question = st.chat_input(
    placeholder="Ask a question about your documents...",
    disabled=not st.session_state.docs_processed
)

# ── Handle new question ──
if question:

    # Show user bubble immediately — don't wait for answer
    with st.chat_message("user"):
        st.write(question)

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                answer, sources = ask_question(question)
                st.write(answer)
                render_source_expander(sources)

            except Exception as e:
                answer = "❌ Something went wrong. Please try again."
                sources = []
                st.error(f"Error details: {str(e)}")

    # Save to history AFTER rendering
    # (prevents double-render in the history loop above)
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "sources": sources
    })