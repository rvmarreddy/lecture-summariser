import os
import json
from pathlib import Path
from typing import List

import numpy as np

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BOOKS_DIR = os.getenv("BOOKS_DIR", "books")
INDEX_DIR = os.getenv("BOOK_INDEX_DIR", "book_index")
BOOK_SUFFIXES = (".txt", ".md", ".pdf", ".epub")

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in (".pdf", ".epub"):
        import fitz  # PyMuPDF opens PDF and EPUB
        doc = fitz.open(str(path))
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        return text
    return ""


def _is_prose(chunk: str) -> bool:
    """Reject table-of-contents / index / page-number junk so it never reaches retrieval."""
    tokens = chunk.split()
    if not tokens:
        return False
    real_words = sum(1 for t in tokens if sum(c.isalpha() for c in t) >= 3)
    digit_ratio = sum(c.isdigit() for c in chunk) / len(chunk)
    return real_words / len(tokens) >= 0.6 and digit_ratio < 0.15


def chunk_text(text: str, words_per_chunk: int = 180, overlap: int = 40) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, words_per_chunk - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + words_per_chunk])
        if len(chunk.split()) >= 20:
            chunks.append(chunk)
        if start + words_per_chunk >= len(words):
            break
    return chunks


def build_index(books_dir: str = BOOKS_DIR, index_dir: str = INDEX_DIR) -> int:
    """Read every book, chunk it, embed the chunks, and persist the index."""
    books = Path(books_dir)
    books.mkdir(exist_ok=True)
    files = [p for p in sorted(books.rglob("*")) if p.suffix.lower() in BOOK_SUFFIXES]
    if not files:
        print(f"No books found in '{books_dir}/'. Add .pdf/.epub/.txt/.md files and re-run.")
        return 0

    passages = []
    for f in files:
        text = _read_document(f)
        for chunk in chunk_text(text):
            if _is_prose(chunk):
                passages.append({"text": chunk, "source": f.name})

    if not passages:
        print("No extractable text found in the provided books.")
        return 0

    print(f"Embedding {len(passages)} passages from {len(files)} book(s)...")
    vectors = _get_embedder().encode(
        [p["text"] for p in passages],
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    out = Path(index_dir)
    out.mkdir(exist_ok=True)
    np.save(out / "embeddings.npy", vectors)
    (out / "passages.json").write_text(json.dumps(passages), encoding="utf-8")
    print(f"Saved index with {len(passages)} passages to '{index_dir}/'.")
    return len(passages)


class BookRetriever:
    """Cosine-similarity search over the persisted book index."""

    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.vectors = None
        self.passages = []
        emb = self.index_dir / "embeddings.npy"
        pas = self.index_dir / "passages.json"
        if emb.exists() and pas.exists():
            self.vectors = np.load(emb)
            self.passages = json.loads(pas.read_text(encoding="utf-8"))

    @property
    def ready(self) -> bool:
        return self.vectors is not None and len(self.passages) > 0

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        if not self.ready or not query.strip():
            return []
        q = _get_embedder().encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores = self.vectors @ q[0]
        k = min(k, len(self.passages))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [self.passages[i]["text"] for i in top]


def load_retriever(index_dir: str = INDEX_DIR):
    """Return a ready retriever, or None when no index has been built yet."""
    r = BookRetriever(index_dir)
    return r if r.ready else None


if __name__ == "__main__":
    build_index()
