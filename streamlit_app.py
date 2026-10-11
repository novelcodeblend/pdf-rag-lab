"""
Streamlit frontend for the classroom PDF RAG lab.
Run: streamlit run streamlit_app.py
"""
import os
import streamlit as st
from rag_core import SimplePDFRAG

st.set_page_config(page_title="PDF RAG Lab", page_icon="📚", layout="wide")

st.title("📚 PDF Question Answering — RAG Lab")
st.caption("Upload a PDF, index its text, and ask questions grounded in the document.")

with st.sidebar:
    st.header("Model settings")
    ollama_url = st.text_input(
        "Ollama URL",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="For local use, keep the default. A cloud-hosted Streamlit app cannot access your computer's localhost.",
    )
    model = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    chunk_size = st.slider("Chunk size (characters)", 400, 1600, 900, 100)
    overlap = st.slider("Chunk overlap (characters)", 0, 300, 150, 25)
    top_k = st.slider("Retrieved chunks", 1, 8, 4)
    temperature = st.slider("Answer creativity (temperature)", 0.0, 1.0, 0.2, 0.1)
    st.divider()
    st.markdown("**Local model setup**")
    st.code("ollama pull llama3.2:3b\nollama serve", language="bash")

if "rag" not in st.session_state:
    st.session_state.rag = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "history" not in st.session_state:
    st.session_state.history = []

uploaded = st.file_uploader("1. Upload a text-based PDF", type=["pdf"])

col1, col2 = st.columns([1, 2])
with col1:
    if st.button("2. Index PDF", type="primary", disabled=uploaded is None, use_container_width=True):
        try:
            rag = SimplePDFRAG(
                ollama_base_url=ollama_url,
                model=model,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            with st.spinner("Reading PDF and building the searchable index..."):
                stats = rag.index_pdf(uploaded.getvalue())
            st.session_state.rag = rag
            st.session_state.doc_name = uploaded.name
            st.session_state.history = []
            st.success("PDF indexed successfully.")
            st.write(f"**Pages with text:** {stats['pages_with_text']}")
            st.write(f"**Chunks created:** {stats['chunks']}")
            st.write(f"**Characters extracted:** {stats['characters']:,}")
        except Exception as exc:
            st.error(f"Could not index this PDF: {exc}")

with col2:
    st.info(
        "RAG does not retrain model weights. This lab extracts PDF text, creates "
        "chunks, retrieves relevant passages, and gives those passages to the LLM."
    )

if st.session_state.rag is not None:
    st.success(f"Active document: {st.session_state.doc_name}")
    question = st.text_input("3. Ask a question about your PDF", placeholder="What are the main points in this document?")
    if st.button("Ask question", use_container_width=False, disabled=not bool(question.strip())):
        try:
            with st.spinner("Retrieving relevant passages and generating an answer..."):
                result = st.session_state.rag.ask(
                    question.strip(), top_k=top_k, temperature=temperature
                )
            st.session_state.history.insert(0, {
                "question": question.strip(),
                "answer": result["answer"],
                "sources": result["sources"],
            })
        except Exception as exc:
            st.error(
                f"Could not generate an answer: {exc}\n\n"
                "For local use, make sure Ollama is running and the selected model has been downloaded."
            )

    for i, item in enumerate(st.session_state.history):
        st.subheader(f"Question: {item['question']}")
        st.write(item["answer"])
        with st.expander("View retrieved source passages"):
            for source_num, source in enumerate(item["sources"], start=1):
                st.markdown(
                    f"**Source {source_num} — page {source['page']} — "
                    f"similarity {source['score']:.3f}**"
                )
                st.write(source["text"])
                st.divider()
else:
    st.markdown("### How to use")
    st.markdown(
        "1. Upload a PDF.\n"
        "2. Click **Index PDF**.\n"
        "3. Enter a question and click **Ask question**.\n"
        "4. Open the source passages to inspect what the retriever supplied to the model."
    )

st.divider()
st.caption("Teaching note: this starter uses TF-IDF retrieval for simplicity. Students can later replace it with embeddings and a vector database.")
