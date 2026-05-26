import tempfile

import streamlit as st

from extract_slides import extract_slide_content
from expand_notes import expand_slide_content, save_markdown
from book_retrieval import load_retriever

st.set_page_config(page_title="Lecture Notes Generator", page_icon="🧠", layout="wide")
st.title("🧠 Lecture Notes Generator (local)")

retriever = load_retriever()
if retriever is None:
    st.info(
        "No book index found. Drop textbooks into `books/` and run `python book_retrieval.py` "
        "to enable grounding. Generation still works without it."
    )
else:
    st.caption(f"📚 Book index loaded — {len(retriever.passages)} passages available for grounding.")

st.subheader("Upload slides (PDF) and optional transcript")
col1, col2 = st.columns(2)
with col1:
    pdf_file = st.file_uploader("Slides PDF", type=["pdf"])
with col2:
    transcript_file = st.file_uploader("Transcript (.txt, optional)", type=["txt"])

manual_transcript = st.text_area("Or paste transcript (optional)", height=120)
page_title = st.text_input("Notes title", placeholder="e.g., Lecture 5 – Reinforcement Learning")

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
        pdf_path = tmp.name

    with st.status("Working...", expanded=True) as status:
        st.write("🔍 Extracting slides...")
        slides = extract_slide_content(pdf_path)
        total = len(slides)
        st.write(f"✅ {total} slides extracted.")

        st.write("🧠 Generating notes with local models (first run downloads weights)...")
        progress = st.progress(0.0)
        expanded = []
        for i, slide in enumerate(slides, start=1):
            expanded.append(
                expand_slide_content([slide], retriever=retriever, extra_transcript=transcript_text)[0]
            )
            progress.progress(i / total, text=f"Slide {i}/{total}")

        out_base = (page_title.strip().replace(" ", "_") or "lecture_notes")
        md_path = save_markdown(expanded, output_path=out_base + ".md")

        st.write("📄 Compiling PDF...")
        pdf_path = None
        try:
            from export_pdf import to_pdf
            combined = "\n\n".join(n["markdown"] for n in expanded)
            pdf_path = to_pdf(combined, out_path=out_base + ".pdf", title=page_title.strip() or "Lecture Notes")
        except Exception as e:
            st.warning(f"PDF export failed: {e}")
        status.update(label="Done!", state="complete")

    with open(md_path, "rb") as f:
        md_bytes = f.read()
    st.download_button("📥 Download Markdown", data=md_bytes, file_name=out_base + ".md", mime="text/markdown")
    if pdf_path:
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download PDF", data=f.read(), file_name=out_base + ".pdf", mime="application/pdf")
    with st.expander("Preview notes"):
        st.markdown(md_bytes.decode("utf-8"))
