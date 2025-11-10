import os
import re
import json
import random
from pathlib import Path
from dotenv import load_dotenv

# === LOAD ENVIRONMENT VARIABLES ===
load_dotenv()

# === CONFIGURATION ===
DATASET_FOLDER = "lecture_notes_dataset"  # Folder containing .md notes
OUTPUT_FILE = "training_data.jsonl"

# === SYSTEM PROMPT (updated for Notion Markdown format) ===
SYSTEM_PROMPT = (
    "You are a specialised model trained to rewrite lecture transcripts and slide content into structured, "
    "Notion-compatible Markdown notes for technical subjects. Follow these formatting and style rules exactly:\n"
    "\n"
    "1. Use Markdown headings consistently to show hierarchy:\n"
    "   - `#` for lecture title (e.g., '# W1 L2: Linear Regression')\n"
    "   - `##` for main sections (e.g., '## Simple regression example')\n"
    "   - `###` for subsections.\n"
    "\n"
    "2. At the top, include any relevant metadata (e.g., lecture date, linked PDF):\n"
    "   Example: `[Week1Lec2LinearRegression.pdf](W1%20L2%20Linear%20Regression/Week1Lec2LinearRegression.pdf)`\n"
    "\n"
    "3. Maintain Notion-style <aside> callouts for side notes, examples, or definitions.\n"
    "   Keep existing emojis if present but do not add new ones.\n"
    "\n"
    "4. Preserve LaTeX equations:\n"
    "   - Inline math: `$...$`\n"
    "   - Display math: `$$...$$`\n"
    "   Example:\n"
    "   ```\n"
    "   $$\n"
    "   w_{t+1} = w_{t} - learning_{rate} * dw\n"
    "   $$\n"
    "   ```\n"
    "\n"
    "5. Preserve image links as-is:\n"
    "   - Example: `![Screenshot 2025-10-03.png](W1%20L2%20Linear%20Regression/Screenshot_2025-10-03.png)`\n"
    "\n"
    "6. Use bullet points and numbered lists where appropriate.\n"
    "7. Use **bold** for emphasis on key terms and variables.\n"
    "8. Keep code fences for code snippets with language tags (e.g., ```python, ```bash).\n"
    "9. Leave one blank line between major sections for readability.\n"
    "10. The final Markdown must render correctly in Notion (supported syntax only).\n"
    "\n"
    "Your goal: produce compact, structured, professional lecture notes without losing technical precision."
)

# === CLEANING HELPERS ===
def clean_markdown(text: str) -> str:
    """Normalize markdown and clean artifacts for training consistency."""
    # Remove base64 inline images
    text = re.sub(r'!\[.*?\]\(data:image\/[a-zA-Z]+;base64,[^)]+\)', '', text)
    # Trim trailing spaces
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse excessive newlines (>2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Ensure spacing before images
    text = re.sub(r'\n{0,1}!\[', '\n\n![', text)
    # Remove stray invisible characters
    text = text.replace('\u200b', '')
    return text.strip()


def collect_md_files(folder: str):
    """Return all markdown files in folder (sorted)."""
    p = Path(folder)
    if not p.exists():
        print(f"❌ Dataset folder '{folder}' not found.")
        return []
    files = sorted(p.glob("*.md"))
    if not files:
        print(f"⚠️ No markdown files found in '{folder}'.")
    return files


def make_jsonl(md_files):
    """Convert markdown notes to OpenAI fine-tuning JSONL format."""
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
    """Show a preview of random examples from the dataset."""
    print("\n🔍 Preview of dataset examples:\n" + "-" * 60)
    if not pairs:
        print("❌ No data to preview.")
        return
    for sample in random.sample(pairs, min(n, len(pairs))):
        assistant_text = sample["messages"][2]["content"]
        print("\n📄 Example:\n" + assistant_text[:300] + ("..." if len(assistant_text) > 300 else ""))
    print("-" * 60)
    print("✅ Preview complete.\n")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    print(f"📁 Scanning '{DATASET_FOLDER}' for Markdown files...")
    md_files = collect_md_files(DATASET_FOLDER)
    if not md_files:
        print("❌ No Markdown files found. Check your dataset folder.")
        raise SystemExit(1)

    print(f"📄 Found {len(md_files)} Markdown files.")
    dataset = make_jsonl(md_files)
    preview_examples(dataset)

    print("🎯 Next: validate with")
    print(f"   openai tools fine_tunes.prepare_data -f {OUTPUT_FILE}")