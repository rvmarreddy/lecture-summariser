import os
import re
import json
from pathlib import Path

from tqdm import tqdm

from book_retrieval import load_retriever
from expand_notes import expand_concepts

DATASET_FOLDER = os.getenv("DATASET_FOLDER", "training")
OUTPUT_FILE = os.getenv("TRAIN_FILE", "training_data.jsonl")


def clean_markdown(text: str) -> str:
    text = re.sub(r'!\[.*?\]\(data:image\/[a-zA-Z]+;base64,[^)]+\)', '', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace('​', '')
    # Drop vestigial Notion <aside> callout tags; keep their inner text as plain Markdown.
    text = re.sub(r'</?aside>', '', text)
    return text.strip()


def collect_md_files(folder: str):
    p = Path(folder)
    if not p.exists():
        print(f"Dataset folder '{folder}' not found.")
        return []
    return sorted(p.rglob("*.md"))


def split_into_sections(md: str):
    """Split a note into sections at ## / ### headings, keeping each heading with its body."""
    sections, current = [], []
    for line in md.split("\n"):
        if re.match(r'^#{1,3}\s', line) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [s for s in sections if len(s.split()) >= 25]


def strip_formatting(section: str) -> str:
    """Reduce a styled section to plain content, a proxy for raw slide text."""
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', section)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'</?aside>', '', text)
    text = re.sub(r'`{1,3}[a-zA-Z]*', '', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def build_training_pairs(md_files, retriever):
    """For each note section: back-expand its plain content into verbose prose, then
    pair (verbose -> original styled section) so the condenser learns the reverse map."""
    pairs = []
    for md in tqdm(md_files, desc="Notes"):
        content = clean_markdown(md.read_text(encoding="utf-8"))
        for section in split_into_sections(content):
            raw = strip_formatting(section)
            if len(raw.split()) < 20:
                continue
            verbose = expand_concepts(raw, retriever=retriever)
            if not verbose.strip():
                continue
            pairs.append({"input": verbose, "output": section})
    return pairs


def save_jsonl(pairs, output_file=OUTPUT_FILE):
    with open(output_file, "w", encoding="utf-8") as f:
        for p in pairs:
            json.dump(p, f)
            f.write("\n")
    print(f"Saved {len(pairs)} training pairs to {output_file}")


if __name__ == "__main__":
    md_files = collect_md_files(DATASET_FOLDER)
    if not md_files:
        raise SystemExit(f"No .md files in '{DATASET_FOLDER}'.")

    print(f"Found {len(md_files)} notes. Building (verbose -> concise) pairs via back-expansion.")
    print("This runs the local model over every section and may take several minutes.")
    retriever = load_retriever()
    if retriever is None:
        print("No book index found — expanding from note content only.")
        print("(Add books to books/ and run `python book_retrieval.py` for textbook grounding.)")
    pairs = build_training_pairs(md_files, retriever)
    save_jsonl(pairs)
    print("Next: python finetune_notes.py")
