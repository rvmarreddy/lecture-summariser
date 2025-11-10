import os
import io
import tempfile
import streamlit as st
from dotenv import load_dotenv
from notion_client_utils import NOTION_FOLDERS, create_page
from extract_slides import extract_slide_content
from expand_notes import expand_slide_content, save_markdown

load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL_ID")  # optional

st.set_page_config(page_title="Lecture → Notes → Notion", page_icon="🧠", layout="wide")
st.title("🧠 Lecture Notes Generator → Notion")

if not NOTION_TOKEN:
    st.error("❌ Missing Notion token! Add `NOTION_TOKEN='your_secret'` to your .env file.")
    st.stop()

# Sidebar: destination
st.sidebar.header("📂 Notion Destination")
folder_names = list(NOTION_FOLDERS.keys())
selected_folder = st.sidebar.selectbox("Select a Notion folder", folder_names)
folder_id = NOTION_FOLDERS[selected_folder]

# Main inputs
st.subheader("Upload Slides (PDF) and optional Transcript")
col1, col2 = st.columns(2)
with col1:
    pdf_file = st.file_uploader("Slides PDF", type=["pdf"])
with col2:
    transcript_file = st.file_uploader("Optional transcript (.txt)", type=["txt"])
manual_transcript = st.text_area("Or paste transcript (optional)", height=120, placeholder="Paste lecture transcript here...")

page_title = st.text_input("📝 New Notion page title", placeholder="e.g., Lecture 5 – Reinforcement Learning")

with st.expander("Advanced"):
    use_web = st.checkbox("Use light web enrichment (Wikipedia)", value=True)
    use_finetuned = st.checkbox("Use fine-tuned model if available", value=bool(FINE_TUNED_MODEL))
    model_override = st.text_input("Model override (optional)", value=FINE_TUNED_MODEL or "")

# Actions
go = st.button("🚀 Generate Notes and Upload to Notion")
status = st.empty()
progress = st.progress(0, text="Idle")

if go:
    if not pdf_file:
        st.warning("Please upload a slides PDF.")
        st.stop()
    if not page_title.strip():
        st.warning("Please enter a Notion page title.")
        st.stop()

    # Prepare transcript (optional)
    transcript_text = ""
    if transcript_file:
        try:
            transcript_text = transcript_file.read().decode("utf-8", errors="ignore")
        except Exception:
            transcript_text = ""
    if manual_transcript.strip():
        transcript_text += ("\n\n" + manual_transcript.strip())

    # Save PDF to a temp file for PyMuPDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_pdf_path = tmp.name

    # Extract slides
    status.markdown("🔍 **Extracting slides...**")
    print("🔍 Extracting slides...")
    slides = extract_slide_content(tmp_pdf_path)
    total = len(slides)
    st.success(f"✅ Extracted {total} slides.")
    print(f"✅ Extracted {total} slides.")
    progress.progress(10, text="Extraction complete")

    # Expand notes slide by slide with live progress
    status.markdown("🧠 **Expanding notes...**")
    print("🧠 Expanding notes...")
    expanded = []
    model_to_use = model_override.strip() or (FINE_TUNED_MODEL if use_finetuned and FINE_TUNED_MODEL else None)

    for i, slide in enumerate(slides, start=1):
        expanded_slide = expand_slide_content(
            [slide],
            model_name=model_to_use,
            web_enrich=use_web,
            extra_transcript=transcript_text
        )[0]
        expanded.append(expanded_slide)
        # Update UI + terminal
        pct = 10 + int(80 * (i / max(total, 1)))
        progress.progress(pct, text=f"Expanding slide {i}/{total}")
        print(f"✅ Expanded slide {i}/{total}")

    # Save combined Markdown and show a download button
    status.markdown("💾 **Saving Markdown...**")
    print("💾 Saving Markdown...")
    md_path = save_markdown(expanded)
    with open(md_path, "rb") as f:
        md_bytes = f.read()
    st.download_button("📥 Download Markdown", data=md_bytes, file_name="lecture_notes.md", mime="text/markdown")
    progress.progress(95, text="Markdown saved")

    # Convert to Notion blocks & upload
    status.markdown("⬆️ **Uploading to Notion...**")
    print("⬆️ Uploading to Notion...")
    try:
        page = create_page(NOTION_TOKEN, folder_id, page_title.strip(), md_bytes.decode("utf-8"))
        page_id = page["id"]
        notion_url = f"https://www.notion.so/{page_id.replace('-', '')}"
        st.success(f"✅ Page created in **{selected_folder}**")
        st.markdown(f"[🔗 Open in Notion]({notion_url})")
        print("✅ Uploaded to Notion:", notion_url)
        progress.progress(100, text="Done")
    except Exception as e:
        st.error(f"❌ Notion upload failed: {e}")
        print("❌ Notion upload failed:", e)