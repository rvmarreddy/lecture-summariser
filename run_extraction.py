import os
import sys
import time
from typing import List, Dict
from extract_slides import extract_slide_content
from expand_notes import expand_slide_content, save_markdown

def run_pipeline(
    pdf_path: str,
    extra_transcript: str = "",
    web_enrich: bool = True,
    model_name: str = None
) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF not found: {pdf_path}")

    print(f"🔍 Extracting slides from: {pdf_path}")
    slides: List[Dict] = extract_slide_content(pdf_path)
    print(f"✅ Extracted {len(slides)} slides\n")

    print("🧠 Expanding slides into enriched notes...\n")

    total = len(slides)
    expanded_notes = []
    for idx, slide in enumerate(slides, start=1):
        sys.stdout.write(f"\r📝 Processing slide {idx}/{total}...")
        sys.stdout.flush()

        expanded_slide = expand_slide_content(
            [slide],
            model_name=model_name,
            web_enrich=web_enrich,
            extra_transcript=extra_transcript
        )[0]
        expanded_notes.append(expanded_slide)

        sys.stdout.write(f"\r✅ Slide {idx}/{total} enriched successfully.\n")
        sys.stdout.flush()
        time.sleep(0.05)

    print("\n💾 Saving all enriched notes...")
    output_file = save_markdown(expanded_notes)

    print(f"\n✅ Notes saved to {output_file}")
    print("📘 Done! You can now import 'lecture_notes.md' into Notion.")
    return output_file

if __name__ == "__main__":
    pdf_path = "test_slides.pdf"
    run_pipeline(pdf_path)