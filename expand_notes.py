import os
from pathlib import Path
from typing import List, Dict

BASE_MODEL = os.getenv("BASE_MODEL", "google/flan-t5-base")
FINETUNED_DIR = os.getenv("FINETUNED_DIR", "finetuned_model")

# Shared between inference (condense_to_style) and training (finetune_notes) so the
# fine-tuned model sees the same instruction it was trained on.
STYLE_INSTRUCTION = (
    "Rewrite the following explanation as concise, well-structured study notes in Markdown. "
    "Use ## and ### headings, define key terms as blockquotes ('> **Term.** ...'), use bullet "
    "and numbered lists, **bold** for key terms, tables where useful, and $...$ / $$...$$ for math. "
    "Keep only the essentials."
)

_tok = None
_base = None
_style_tok = None
_style = None
_device = None


def _get_device():
    global _device
    if _device is not None:
        return _device
    import torch
    if torch.cuda.is_available():
        _device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"
    return _device


def _load_base():
    global _tok, _base
    if _base is not None:
        return
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    _base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(_get_device())
    _base.eval()


def _load_style():
    global _style_tok, _style
    if _style is not None:
        return
    if Path(FINETUNED_DIR).exists():
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _style_tok = AutoTokenizer.from_pretrained(FINETUNED_DIR)
        _style = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_DIR).to(_get_device())
        _style.eval()
    else:
        # No fine-tuned model yet: fall back to the base model with the style prompt.
        _load_base()
        _style_tok, _style = _tok, _base


def _generate(model, tok, prompt: str, max_new_tokens: int, min_new_tokens: int = 0,
              length_penalty: float = 1.0, max_input_tokens: int = 512) -> str:
    import torch
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(_get_device())
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=4,
            length_penalty=length_penalty,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tok.decode(out[0], skip_special_tokens=True).strip()


def expand_concepts(text: str, retriever=None, transcript: str = "", k: int = 3) -> str:
    """Stage 1: explain the slide's concepts in detail, grounded in retrieved book passages."""
    _load_base()
    context = ""
    if retriever is not None and text.strip():
        passages = retriever.retrieve(text, k=k)
        if passages:
            # Trim each passage so the context survives the input-token budget instead of
            # being truncated off the end of the prompt.
            context = "\n".join(f"- {' '.join(p.split()[:90])}" for p in passages)

    prompt = (
        "You are a teacher. Explain the concepts below in clear, detailed prose so a student "
        "understands them. Define key terms, give intuition, and explain any formulas.\n\n"
        f"Concepts:\n{text or '[no text]'}\n"
    )
    if transcript.strip():
        prompt += f"\nLecture transcript context:\n{transcript[:600]}\n"
    if context:
        prompt += f"\nReference material from textbooks:\n{context}\n"
    prompt += "\nDetailed explanation:"
    return _generate(_base, _tok, prompt, max_new_tokens=320, min_new_tokens=100,
                     length_penalty=1.4, max_input_tokens=768)


def condense_to_style(verbose_text: str) -> str:
    """Stage 2: rewrite the verbose explanation into concise notes in the learned style."""
    _load_style()
    prompt = f"{STYLE_INSTRUCTION}\n\nExplanation:\n{verbose_text}\n\nNotes:"
    return _generate(_style, _style_tok, prompt, max_new_tokens=256, min_new_tokens=40,
                     length_penalty=1.1)


def expand_slide_content(slide_data: List[Dict], retriever=None, extra_transcript: str = "") -> List[Dict]:
    """Per-slide pipeline: expand concepts then condense to style. Emits topic-structured
    Markdown (no 'Slide N' heading); emitting extracted slide images is a deferred extension."""
    notes = []
    for entry in slide_data:
        verbose = expand_concepts(entry.get("text", ""), retriever=retriever, transcript=extra_transcript)
        styled = condense_to_style(verbose)
        notes.append({"slide": entry["slide"], "markdown": styled})
    return notes


def save_markdown(expanded_notes: List[Dict], output_path: str = "lecture_notes.md") -> str:
    with open(output_path, "w", encoding="utf-8") as f:
        for note in expanded_notes:
            f.write(note["markdown"])
            f.write("\n\n")
    return output_path
