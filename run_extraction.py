import os
import re
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from extract_slides import extract_slide_content
from book_retrieval import load_retriever, TextRetriever
from note_writer import (content_points, review_points, structure_points,
                          dedup_blocks, dedup_sections, verify_relevant, _fix_box_titles, _tidy_headings)

# Minimum slide<->passage cosine similarity for a book passage to be used as grounding. Below this a
# passage is treated as off-topic and dropped (stops thin-slide topics pulling tangential book content).
BOOK_MIN_SCORE = float(os.getenv("BOOK_MIN_SCORE", "0.42"))
from export_pdf import latex_to_pdf, timestamped_output


def _chunk_slides(slides: List[Dict], budget: int = 4000) -> List[List[str]]:
    """Group consecutive slides into ~budget-char groups, returning each group's per-slide texts
    (so retrieval can query slide-by-slide while generation still sees the whole group)."""
    groups, cur, cur_len = [], [], 0
    for s in slides:
        t = s.get("text", "").strip()
        if not t:
            continue
        if cur and cur_len + len(t) + 1 > budget:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(t)
        cur_len += len(t) + 1
    if cur:
        groups.append(cur)
    return groups or [[]]


def _is_boilerplate(slide_text: str) -> bool:
    """Skip title / syllabus / agenda slides: their text makes a poor retrieval query and would
    ground the notes in unrelated passages (e.g. a week-by-week course outline)."""
    t = slide_text.strip()
    if len(t) < 40:
        return True
    low = t.lower()
    return low.count("week ") >= 3 or low.count("lecture ") >= 3


def _ground(retriever, slide_texts: List[str], k_per_slide: int = 2, k_max: int = 6,
            min_score: float = 0.0, topic: str = None) -> str:
    """Retrieve grounding passages slide-by-slide (skipping boilerplate), deduped, so every topic in
    the group is grounded rather than only the group's first ~1000 chars (the old single-query bug).
    `min_score` drops weakly-related passages; if `topic` is given, a local-LLM scope gate then keeps only
    passages that genuinely explain that topic (fixes the bi-encoder's surface-word mis-attributions)."""
    seen, out = set(), []
    for st in slide_texts:
        if _is_boilerplate(st):
            continue
        for p in retriever.retrieve(st, k=k_per_slide, min_score=min_score):
            key = p[:80]
            if key not in seen:
                seen.add(key)
                out.append(p)
        if len(out) >= k_max:
            break
    passages = out[:k_max]
    if topic and passages:
        passages = verify_relevant(topic, passages)
    return "\n\n".join(passages)


_VEC_RE = re.compile(r"\[\s*-?\d[\d.]*(?:\s*,\s*-?\d[\d.]*)+\s*\]")


def _num_anchor(texts: List[str], cap: int = 12) -> List[str]:
    """Pull the slides' literal numeric content — bracket vectors like [1,0,0,0] and runs of number-only
    lines (one-hot grids, similarity values) — so the model reuses these exact values instead of inventing
    its own example (the math-preservation gap: e.g. substituting a textbook example for the slide's matrix)."""
    found = []
    for t in texts:
        found += _VEC_RE.findall(t)
        run = []
        for line in t.splitlines() + [""]:
            s = line.strip()
            if s and re.fullmatch(r"[-\d.\s]+", s) and any(ch.isdigit() for ch in s):
                run.append(s)
            else:
                if len(run) >= 3:
                    multi = [r for r in run if len(re.findall(r"-?\d[\d.]*", r)) >= 2]
                    if multi:
                        # A grid: one item per multi-number row, so each row can dedup against its bracket form.
                        found.extend(multi)
                    else:
                        # A vertical column of lone values is ONE vector printed one element per line
                        # (the slides' TF-IDF layout) — join it; per-row it would be meaningless singletons.
                        found.append(" ".join(run))
                run = []
    seen, out = set(), []
    for v in found:
        v = " ".join(v.split())
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out[:cap]


_BRACKET_RE = re.compile(r"\[[^\[\]\n]+\]")


def _extract_examples(texts: List[str], cap: int = 10, seen: set = None) -> List[str]:
    """Pull the slide's OWN worked examples verbatim — token lists / vectors like [@uniofbath, is, ...]
    or [1, 0, 0, 0], explicit 'X -> Y' / 'X becomes Y' transforms, and number grids — so they can be PLACED
    verbatim instead of re-authored by the model (which is where it swaps in off-slide examples). A shared
    `seen` set deduplicates examples ACROSS topics, so the same token list isn't placed in several sections."""
    items = []
    for t in texts:
        for b in _BRACKET_RE.findall(t):
            if "," in b or any(ch.isdigit() for ch in b):
                items.append(" ".join(b.split()))
        for line in t.splitlines():
            s = " ".join(line.split())
            if 3 < len(s) < 200 and ("->" in s or "→" in s or " becomes " in s) and ('"' in s or '“' in s or "[" in s):
                items.append(s)
    items += _num_anchor(texts, cap=12)   # generous inner cap: dedup/subsumption + the final [:cap] bound it
    core = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    if seen is None:
        seen = set()
    out = []
    for v in items:
        if v and len(v) > 2 and (core(v) or v) not in seen and v not in out:
            out.append(v)
    # Drop items subsumed by a longer one — INCLUDING equal cores: "[111000000]", "1 1 1 0 0 0 0 0 0" and
    # "[1, 1, 1, 0, 0, 0, 0, 0, 0]" are the same vector three ways; keep only the longest/labelled form.
    keep = []
    for v in sorted(out, key=len, reverse=True):
        cv = core(v)
        if cv and not any(cv in core(k) for k in keep):
            keep.append(v)
    kept = [v for v in out if v in keep][:cap]
    for v in kept:                      # cores (not verbatim strings) into the cross-topic seen set,
        seen.add(core(v) or v)          # so a format variant can't re-place the vector in a later topic
    return kept


def _tex_verbatim(s: str) -> str:
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"),
                 ("#", "\\#"), ("$", "\\$"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"),
                 ("^", "\\textasciicircum{}")):
        s = s.replace(a, b)
    return s


_NUMVEC = re.compile(r"\[\s*-?\d[\d.]*(?:\s*,\s*-?\d[\d.]*)*\s*\]")


def _fmt_item(v: str) -> str:
    """Format one verbatim slide item as proper LaTeX (vectors -> math, token lists -> typewriter), instead
    of dumping it as raw monospace text (the judges' main complaint about the appended-examples version)."""
    s = " ".join(v.split())
    m = re.match(r"^(.+?)\s*(?:->|→)\s*(\[[\d ,.\-]+\])$", s)          # 'London -> [1, 0, 0, 0]'
    if m and _NUMVEC.fullmatch(m.group(2).replace(" ", "")):
        return "$\\text{%s} \\to %s$" % (_tex_verbatim(m.group(1)), m.group(2).replace(" ", ""))
    if _NUMVEC.fullmatch(s.replace(" ", "")):                          # '[1, 0, 0, 0]'
        return "$%s$" % s.replace(" ", "")
    if re.fullmatch(r"[-\d.\s]+", s):                                  # bare number row '1 0 0 0'
        return "$[%s]$" % ", ".join(s.split())
    if s.startswith("[") and s.endswith("]"):                         # token list '[@uniofbath, is, ...]'
        return "\\texttt{%s}" % _tex_verbatim(s)
    return _tex_verbatim(s)


def _stem_table(texts: List[str]) -> str:
    """Detect a 3-column word table (Original/Stemmed/Lemmatised) in the slides and render it as a real
    tabularx, so the stem/lemma comparison is a table instead of flat run-on lines."""
    for t in texts:
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        low = [l.lower() for l in lines]
        for i in range(len(lines) - 2):
            if "stem" in low[i + 1] and "lemmat" in low[i + 2]:
                rows, j = [], i + 3
                while j + 3 <= len(lines):
                    trio = lines[j:j + 3]
                    if all(re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", w) for w in trio):
                        rows.append(trio)
                        j += 3
                    else:
                        break
                if len(rows) >= 2:
                    body = " \\\\\n".join(" & ".join(_tex_verbatim(w) for w in r) for r in rows)
                    return ("\\begin{center}\\begin{tabularx}{0.8\\linewidth}{lXX}\n"
                            "\\textbf{Word} & \\textbf{Stemmed} & \\textbf{Lemmatised} \\\\\n\\midrule\n"
                            + body + " \\\\\n\\end{tabularx}\\end{center}\n")
    return ""


def _sim_matrix(texts: List[str]) -> str:
    """Detect a small labelled similarity matrix (single-letter row labels + an n x n grid of decimals,
    e.g. the slide-39 a/b/c cosine matrix) and render it as a bordered array — so the model doesn't draw a
    broken/truncated one. Distinguishes the matrix from one-hot vectors by requiring a non-integer value."""
    lab = re.compile(r"^[a-z]$")
    for t in texts:
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        rows, cur_lab, cur = [], None, []
        for l in lines:
            if lab.fullmatch(l):
                if cur_lab is not None and cur:
                    rows.append((cur_lab, cur))
                cur_lab, cur = l, []
            elif cur_lab is not None and re.fullmatch(r"[\d.\s]+", l):
                cur += re.findall(r"\d+(?:\.\d+)?", l)
            elif cur_lab is not None:                    # a non-number, non-label line ends the matrix region
                if cur:
                    rows.append((cur_lab, cur))
                cur_lab, cur = None, []
        if cur_lab is not None and cur:
            rows.append((cur_lab, cur))
        n = len(rows)
        sq = [r for r in rows if len(r[1]) == n]
        if n >= 2 and len(sq) == n and any("." in v for _, vals in sq for v in vals):
            labs = [r[0] for r in sq]
            header = " & " + " & ".join(labs) + " \\\\ \\hline\n"
            body = " \\\\\n".join(r[0] + " & " + " & ".join(r[1]) for r in sq)
            return "\\[\n\\begin{array}{c|%s}\n%s%s \\\\\n\\end{array}\n\\]" % ("c" * n, header, body)
    return ""


def _protected_block(texts: List[str], seen: set = None, topic: str = "") -> str:
    """A dobox of the slide's worked examples, formatted (vectors as math, a real stem/lemma table, the
    similarity matrix as an array) and placed verbatim — the protected content the model must NOT author.
    `seen` deduplicates the example items across topics so a token list isn't placed in several sections.
    `topic` qualifies the box title so ten traces don't all carry the identical anonymous title."""
    items = _extract_examples(texts, seen=seen)
    table = _stem_table(texts)
    matrix = _sim_matrix(texts)
    if not items and not table and not matrix:
        return ""
    parts = []
    if items:
        parts.append("\\begin{itemize}\n" + "\n".join("  \\item " + _fmt_item(v) for v in items)
                     + "\n\\end{itemize}")
    if matrix:
        parts.append(matrix)
    if table:
        parts.append(table)
    label = "from the slides: %s" % _tex_verbatim(topic[:40]) if topic else "from the slides"
    return ("\n\n\\begin{dobox}[%s]\n" % label
            + "\n".join(parts) + "\n\\end{dobox}\n")


_TITLE_RE = re.compile(r"\\(?:sub)?section\*?\{([^}]*)\}")


def _section_titles(body: str) -> List[str]:
    """Section/subsection titles in a generated body, so later parts know what's already covered."""
    seen, out = set(), []
    for t in _TITLE_RE.findall(body):
        t = t.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def _slide_topic(text: str) -> tuple:
    """Best-guess (section_prefix, topic) for a slide. Prefer a 'Section: Subtitle' header (e.g.
    'Document Processing: Normalisation' -> ('Document Processing', 'Normalisation')); otherwise the
    slide's title line with an empty prefix. URLs and sentence-colons are ignored so they don't become
    spurious topics."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines:
        if "://" in l or l.lower().startswith("http"):
            continue
        if ": " in l and len(l) < 60 and not l[0].isdigit():
            head, tail = (p.strip() for p in l.split(": ", 1))
            if tail and len(tail.split()) <= 6:       # a real short subtitle, not a sentence
                return head, tail
    for l in lines:                                    # fallback: first real title line
        if "://" in l or l.lower().startswith("http") or len(l) < 2:
            continue
        return "", l
    return "", "Notes"


_SKIP_TOPICS = {"learning objectives", "feedback", "where are we?", "where are we",
                "document processing"}  # the bare prefix = recap/navigation slides; each stage has its own topic
_EMOJI_RE = re.compile(r"emoji_\w+")  # lecturer's placeholder for an emoji; the '_' breaks LaTeX, so strip it


def _clean_slide_text(text: str) -> str:
    return _EMOJI_RE.sub("(emoji)", text)


_VAGUE_TAIL_RE = re.compile(r"(?:the|a|an)\s+\w+", re.I)
_VAGUE_TAILS = {"motivation", "overview", "introduction", "intro", "recap", "summary",
                "example", "examples", "the problem", "the concept", "the idea", "the solution"}
_MERGE_CAP = 6000   # max combined chars when merging same-prefix neighbours (content stage caps at 12000)


def _vague_tail(tail: str) -> bool:
    """A subtitle too generic (or too junky) to stand as a section title on its own — e.g. 'The Problem',
    'The Concept', or a raw set like '{off, on, lights}'. These merge into / get qualified by their
    section prefix instead of becoming standalone vague sections."""
    t = " ".join(tail.lower().split())
    return (t in _VAGUE_TAILS or _VAGUE_TAIL_RE.fullmatch(t) is not None
            or not tail[:1].isalpha())


def _topic_groups(slides: List[Dict]) -> List[tuple]:
    """Plan an outline: group slides by their 'Section: Subtitle' header into (topic, [slide_texts]) so each
    lecture topic becomes one focused generation unit. This mimics how a strong writer plans the document
    before writing — every topic gets its own section, so none can be silently dropped, and there is no
    arbitrary char-chunk boundary cutting a topic in half. Syllabus/title/objectives/feedback slides are
    dropped. Adjacent groups sharing a section prefix are merged when a subtitle is vague — so one slide
    theme ('Frequency Representations: The Problem' / ': Positional Encoding') is no longer split across
    sections — and a standalone vague subtitle is qualified by its prefix; specific subtitles keep their
    own sections."""
    order, by_key = [], {}
    for s in slides:
        t = _clean_slide_text(s.get("text", "").strip())
        if not t:
            continue
        prefix, topic = _slide_topic(t)
        k = (prefix.lower(), topic.lower())
        if k not in by_key:
            by_key[k] = (prefix, topic, [])
            order.append(k)
        by_key[k][2].append(t)
    groups = []
    for k in order:
        prefix, topic, texts = by_key[k]
        if topic.lower() in _SKIP_TOPICS or all(_is_boilerplate(x) for x in texts):
            continue
        groups.append([prefix, [topic], texts])
    # Merge an adjacent same-prefix pair when either subtitle is vague, capped so a deck whose slides all
    # share one prefix (e.g. every L2 slide is 'Document Processing: ...') cannot collapse into a mega-topic.
    merged = []
    for g in groups:
        if merged:
            m = merged[-1]
            same = m[0] and g[0] and m[0].lower() == g[0].lower()
            vague = any(_vague_tail(t) for t in m[1] + g[1])
            fits = sum(len(x) for x in m[2] + g[2]) <= _MERGE_CAP
            if same and vague and fits:
                m[1] += g[1]
                m[2] += g[2]
                continue
        merged.append(g)
    keep = []
    for prefix, tails, texts in merged:
        if len(tails) > 1:                            # a merged theme: the section prefix names it
            topic = prefix
        elif prefix and _vague_tail(tails[0]):        # standalone vague subtitle: qualify it
            topic = f"{prefix}: {tails[0]}" if tails[0][:1].isalpha() else prefix
        else:
            topic = tails[0]
        keep.append((topic, texts))
    return keep


def run_pipeline(pdf_path: str, extra_transcript: str = "", output_name: str = None,
                 title: str = None, chunk_chars: int = 4000) -> str:
    """Slides (+transcript) -> per-topic grounded Qwen calls with coverage enforcement -> LaTeX -> PDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    title = title or Path(pdf_path).stem.replace("_", " ")

    print(f"Extracting slides from: {pdf_path}")
    slides = extract_slide_content(pdf_path)
    topics = _topic_groups(slides)
    print(f"Extracted {len(slides)} slides -> {len(topics)} topics: {', '.join(t for t, _ in topics)}")

    retriever = load_retriever()
    print("Book grounding:", "enabled" if retriever else "disabled (no index)")
    tr = TextRetriever(extra_transcript) if extra_transcript.strip() else None
    if tr:
        print(f"Transcript grounding: {len(tr.chunks)} chunks available")

    # One focused, grounded generation per topic, then verify each topic actually produced a section
    # (coverage enforcement: a topic can no longer be silently dropped).
    bodies, covered, missing, seen_ex = [], [], [], set()
    for i, (topic, texts) in enumerate(topics, start=1):
        book = _ground(retriever, texts, k_per_slide=2, k_max=6, min_score=BOOK_MIN_SCORE, topic=topic) if retriever else ""
        tctx = _ground(tr, texts, k_per_slide=1, k_max=4) if tr else ""
        # Protected-span tags: the slide's worked examples are formatted and placed verbatim where the model
        # puts the [[EXAMPLES]] tag (it must NOT author or edit them) — it only explains + adds textbook.
        # `seen_ex` dedups example items across topics, so a token list is placed in only its first topic.
        examples = _protected_block(texts, seen=seen_ex, topic=topic)
        source = (f"LECTURE SLIDES for the topic \"{topic}\":\n" + "\n".join(texts)
                  + f"\n\nTEXTBOOK CONTEXT:\n{book}")
        # Staged pipeline: 1) CONTENT points (coverage+correctness, examples frozen as [[EXAMPLES]]) ->
        # 2) REVIEW (fix/drop unsupported points) -> 3) STRUCTURE (arrange into styled LaTeX, token kept) ->
        # substitute the protected examples last. Each stage is a small, focused task for the local model.
        print(f"Topic {i}/{len(topics)}: {topic} — content...", end="", flush=True)
        pts = content_points(topic, source, transcript=tctx, avoid_topics=covered[-40:] or None,
                             has_examples=bool(examples))
        print(" review...", end="", flush=True)
        pts = review_points(topic, pts, source)
        print(" structure...", flush=True)
        body = structure_points(topic, pts)
        if body.strip():
            if examples:                                 # substitute the protected token inline (fallback: append)
                if re.search(r"\[\[\s*EXAMPLES\s*\]\]", body):
                    body = re.sub(r"\[\[\s*EXAMPLES\s*\]\]", lambda _m: examples, body, count=1)
                    body = re.sub(r"\[\[\s*EXAMPLES\s*\]\]", "", body)   # drop any extra tags
                else:
                    body = body + examples
            else:
                body = re.sub(r"\[\[\s*EXAMPLES\s*\]\]", "", body)       # no examples: strip any stray token
            bodies.append(body)
            covered += _section_titles(body)
        else:
            missing.append(topic)
    if missing:
        print(f"  WARNING: no output for topics: {', '.join(missing)}")
    joined = "\n\n".join(b for b in bodies if b.strip())
    # Deterministic, lossless dedup: drop only repeated section titles and exact-duplicate callout boxes.
    print("Removing cross-topic repetition (section + box dedup)...")
    body = _tidy_headings(_fix_box_titles(dedup_blocks(dedup_sections(joined))))

    base = str(Path(output_name).with_suffix("")) if output_name else timestamped_output(title.replace(" ", "_"))
    try:
        pdf = latex_to_pdf(body, base + ".pdf", title=title, head_r=title, author="Generated locally")
    except RuntimeError:
        print("Deduped notes failed to compile; falling back to the raw concatenation.")
        pdf = latex_to_pdf(joined, base + ".pdf", title=title, head_r=title, author="Generated locally")
    print(f"\nPDF saved to {pdf}\n.tex saved to {base}.tex")
    return pdf


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "inputs/slides/L1.pdf"
    transcript = ""
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        transcript = open(sys.argv[2], encoding="utf-8", errors="ignore").read()
    run_pipeline(pdf, extra_transcript=transcript)
