import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st

from book_retrieval import load_retriever
from run_extraction import run_pipeline

st.set_page_config(page_title="Lecture Notes Generator", page_icon="📝", layout="wide")
st.title("📝 Lecture Notes Generator (local)")

retriever = load_retriever()
if retriever is None:
    st.info("No book index found. Drop textbooks into `books/` and run `python src/book_retrieval.py` to enable grounding.")
else:
    st.caption(f"📚 Book index loaded — {len(retriever.passages)} passages for grounding.")
st.caption("Generates styled PDF revision notes locally via Qwen2.5-7B (Ollama). Make sure the Ollama server is running.")

col1, col2 = st.columns(2)
with col1:
    pdf_file = st.file_uploader("Slides PDF", type=["pdf"])
with col2:
    transcript_file = st.file_uploader("Transcript (.txt, optional)", type=["txt"])
manual_transcript = st.text_area("Or paste transcript (optional)", height=120)
page_title = st.text_input("Notes title", placeholder="e.g., Handling Sparsity")

if st.button("🚀 Generate notes"):
    if not pdf_file:
        st.warning("Please upload a slides PDF.")
        st.stop()

    transcript_text = ""
    if transcript_file:
        transcript_text = transcript_file.read().decode("utf-8", errors="ignore")
    if manual_transcript.strip():
        transcript_text += "\n\n" + manual_transcript.strip()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_pdf = tmp.name

    with st.status("Generating (extract → ground → Qwen → LaTeX → PDF)…", expanded=True) as status:
        st.write("Running the local pipeline — this takes a couple of minutes.")
        try:
            pdf_path = run_pipeline(tmp_pdf, extra_transcript=transcript_text,
                                    title=page_title.strip() or "Lecture Notes")
            status.update(label="Done!", state="complete")
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Generation failed: {e}")
            st.stop()

    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download PDF", data=f.read(),
                           file_name=os.path.basename(pdf_path), mime="application/pdf")
    tex_path = pdf_path[:-4] + ".tex"
    if os.path.exists(tex_path):
        with open(tex_path, "rb") as f:
            st.download_button("📥 Download LaTeX (.tex)", data=f.read(),
                               file_name=os.path.basename(tex_path), mime="text/plain")
