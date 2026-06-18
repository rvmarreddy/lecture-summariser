"""Eval runner. Generates notes for a set of decks under the CURRENT env (OLLAMA_MODEL, STRUCTURE_MODE,
NLI_ENABLE) and dumps the source-slide text alongside, so a judge can score notes-vs-slides reproducibly.

Usage (from repo root):
    .venv/bin/python eval/run_eval.py <tag> <deck> [<deck> ...]
e.g.
    OLLAMA_MODEL=qwen2.5:7b STRUCTURE_MODE=llm  NLI_ENABLE=0 .venv/bin/python eval/run_eval.py old L3
    OLLAMA_MODEL=qwen2.5:7b STRUCTURE_MODE=json NLI_ENABLE=1 .venv/bin/python eval/run_eval.py new L3

Outputs land in eval/out/: <tag>_<deck>.tex/.pdf (notes) and <deck>_slides.txt (source).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                   # run_extraction.py lives at the repo root
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)                                  # so inputs/ models/ outputs/ relative paths resolve

from extract_slides import extract_slide_content  # noqa: E402
from run_extraction import run_pipeline           # noqa: E402

OUT = ROOT / "eval" / "out"
OUT.mkdir(parents=True, exist_ok=True)
SLIDES_DIR = ROOT / "inputs" / "slides"


def dump_slides(deck: str) -> Path:
    pdf = SLIDES_DIR / f"{deck}.pdf"
    slides = extract_slide_content(str(pdf))
    text = "\n\n".join(f"--- slide {s['slide']} ---\n{s['text']}" for s in slides if s.get("text"))
    p = OUT / f"{deck}_slides.txt"
    p.write_text(text, encoding="utf-8")
    return p


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    tag, decks = sys.argv[1], sys.argv[2:]
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    mode = os.getenv("STRUCTURE_MODE", "json")
    nli = os.getenv("NLI_ENABLE", "1")
    print(f"== eval tag={tag} model={model} structure={mode} nli={nli} decks={decks} ==")
    for deck in decks:
        pdf = SLIDES_DIR / f"{deck}.pdf"
        if not pdf.exists():
            print(f"  SKIP {deck}: {pdf} not found")
            continue
        dump_slides(deck)
        base = OUT / f"{tag}_{deck}"
        print(f"  generating {base.name} ...")
        try:
            run_pipeline(str(pdf), output_name=str(base) + ".pdf", title=f"{deck} ({tag})")
            print(f"  OK -> {base}.tex / {base}.pdf")
        except Exception as e:
            print(f"  FAILED {deck}: {e}")


if __name__ == "__main__":
    main()
