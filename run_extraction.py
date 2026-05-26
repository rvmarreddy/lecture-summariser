import os
import sys
from pathlib import Path
from typing import List, Dict

from extract_slides import extract_slide_content
from expand_notes import expand_slide_content, save_markdown
from export_pdf import to_pdf
from book_retrieval import load_retriever


def run_pipeline(pdf_path: str, extra_transcript: str = "", output_name: str = "lecture_notes.md") -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    retriever = load_retriever()
    print("Book grounding:", "enabled" if retriever else "disabled (no index)")

    print(f"Extracting slides from: {pdf_path}")
    slides: List[Dict] = extract_slide_content(pdf_path)
    total = len(slides)
    print(f"Extracted {total} slides\n")

    expanded = []
    for idx, slide in enumerate(slides, start=1):
        print(f"Processing slide {idx}/{total}...")
        expanded.append(
            expand_slide_content([slide], retriever=retriever, extra_transcript=extra_transcript)[0]
        )

    md_path = save_markdown(expanded, output_path=output_name)
    print(f"\nNotes saved to {md_path}")

    pdf_path = str(Path(output_name).with_suffix(".pdf"))
    try:
        to_pdf("\n\n".join(n["markdown"] for n in expanded), out_path=pdf_path,
               title=Path(output_name).stem.replace("_", " "))
        print(f"PDF saved to {pdf_path}")
    except Exception as e:
        print(f"PDF export failed ({e}); Markdown is still available.")
    return md_path


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "Other/test_slides.pdf"
    run_pipeline(pdf)
