import os
import re
import json
import random
from pathlib import Path
from dotenv import load_dotenv

# === LOAD ENVIRONMENT VARIABLES ===
load_dotenv()

# === CONFIGURATION ===
DATASET_FOLDER = "lecture_notes_dataset"  # Folder with .md lecture notes
OUTPUT_FILE = "training_data.jsonl"

# === SYSTEM PROMPT (improved for Notion-style Markdown) ===
SYSTEM_PROMPT = (
    "You are a specialised model trained to transform lecture slides and transcripts into structured, "
    "Notion-compatible Markdown notes for technical subjects. Follow these exact stylistic and structural rules:\n"
    "\n"
    "1. Use Markdown headings consistently to indicate hierarchy:\n"
    "   - `#` for lecture title (e.g. '# W1 L2: Linear Regression')\n"
    "   - `##` for main section topics (e.g. '## Simple regression example')\n"
    "   - `###` for sub-sections (e.g. '### Process for solving')\n"
    "\n"
    "2. Include relevant metadata (dates, linked PDFs, etc.) at the top when present.\n"
    "   Example: `[Week1Lec2LinearRegression.pdf](W1%20L2%20Linear%20Regression/Week1Lec2LinearRegression.pdf)`\n"
    "\n"
    "3. Maintain callouts using Notion-style <aside> blocks for side notes, examples, or definitions.\n"
    "   Keep any emojis that already exist, but do not add new ones.\n"
    "\n"
    "4. Preserve equations in LaTeX format — `$...$` for inline and `$$...$$` for display math.\n"
    "   Example:\n"
    "   ```\n"
    "   $$\n"
    "   w_{t+1} = w_{t} - learning_{rate} * dw\n"
    "   $$\n"
    "   ```\n"
    "\n"
    "5. Keep image references exactly as in the original Markdown:\n"
    "   - Format: `![Screenshot 2025-10-03 at 10.19.50.png](W1%20L2%20Linear%20Regression/Screenshot_2025-10-03_at_10.19.50.png)`\n"
    "\n"
    "6. Use bullet and numbered lists with proper indentation for structure.\n"
    "\n"
    "7. Maintain inline emphasis using **bold** for key ideas, variables, and terms.\n"
    "\n"
    "8. Preserve code blocks and syntax highlighting (```python, ```bash, etc.).\n"
    "\n"
    "9. End each major section with a blank line; ensure readability and consistent spacing.\n"
    "\n"
    "10. The output must remain valid Markdown and render properly in Notion without extra tags or HTML "
    "(other than <aside> blocks).\n"
    "\n"
    "11. Ensure the final Markdown is directly convertible to Notion blocks using markdown_to_blocks(), "
    "including headings, lists, callouts, quotes, code, and equations."
)

# === HELPERS ===

def clean_markdown(text: str) -> str:
    """Normalize Markdown for consistent fine-tuning."""
    # Remove base64 inline images
    text = re.sub(r'!\[.*?\]\(data:image\/[a-zA-Z]+;base64,[^)]+\)', '', text)
    # Trim trailing spaces
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse >2 blank lines
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)
    # Remove spaces inside math delimiters
    text = re.sub(r'\s*\$\s*', '$', text)
    # Ensure a blank line before images
    text = re.sub(r'\n{0,1}!\[', '\n\n![', text)
    return text.strip()

def collect_md_files(folder: str):
    p = Path(folder)
    if not p.exists():
        print(f"❌ Dataset folder '{folder}' not found.")
        return []
    return sorted(p.glob("*.md"))

def make_jsonl(md_files):
    pairs = []
    for md in md_files:
        content = md.read_text(encoding="utf-8").strip()
        content = clean_markdown(content)
        if not content:
            continue

        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Reformat this lecture into concise Notion-style Markdown notes."},
                {"role": "assistant", "content": content}
            ]
        }
        pairs.append(record)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in pairs:
            json.dump(ex, f)
            f.write("\n")

    print(f"\n✅ Saved {len(pairs)} examples to {OUTPUT_FILE}")
    return pairs

def preview_examples(pairs, n=3):
    print("\n🔍 Preview of dataset examples:\n" + "-" * 60)
    if not pairs:
        print("❌ No data to preview.")
        return
    for sample in random.sample(pairs, min(n, len(pairs))):
        assistant_text = sample["messages"][2]["content"]
        print("\n📄 Example:\n" + assistant_text[:300] + ("..." if len(assistant_text) > 300 else ""))
    print("-" * 60)
    print("✅ Preview complete.\n")

# === MAIN ===

if __name__ == "__main__":
    print(f"📁 Scanning '{DATASET_FOLDER}' for Markdown files...")
    md_files = collect_md_files(DATASET_FOLDER)
    if not md_files:
        print("❌ No Markdown files found. Check the dataset folder.")
        raise SystemExit(1)

    print(f"📄 Found {len(md_files)} Markdown files.")
    dataset = make_jsonl(md_files)
    preview_examples(dataset)

    print("🎯 Next: validate with")
    print(f"   openai tools fine_tunes.prepare_data -f {OUTPUT_FILE}")