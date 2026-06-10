import os
import re
import json
import urllib.request
from pathlib import Path

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        inner = t.split("```")
        t = inner[1] if len(inner) >= 2 else t.strip("`")
        if t.lstrip().lower().startswith("latex"):
            t = t.lstrip()[5:]
    return t.strip()


# Math environments (where '_', '^', '&' are all legitimate) and table environments
# (text-mode, but '&' is a column separator). Tracked so we only escape specials in prose.
_MATH_ENVS = {
    "equation", "equation*", "displaymath", "math",
    "align", "align*", "aligned", "alignat", "alignat*", "alignedat",
    "gather", "gather*", "multline", "multline*", "flalign", "flalign*",
    "eqnarray", "eqnarray*", "array", "cases", "dcases", "split",
    "matrix", "bmatrix", "pmatrix", "vmatrix", "Vmatrix", "Bmatrix", "smallmatrix",
}
_TABLE_ENVS = {"tabular", "tabularx", "tabular*"}
_BE_RE = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")
_CURRENCY = re.compile(r"\$(\d[\d,.]*)")   # a literal currency amount like $1 or $37.42 (not a math span)


def _sanitize_body(text: str) -> str:
    """Clean model output so xelatex compiles: strip code fences and any leaked preamble, then escape
    stray LaTeX specials that appear in prose. '&' is kept inside tables and math; '_', '#', '%' are kept
    only inside math ($...$, \\[ \\], or a math environment) and escaped everywhere else — so a token like
    emoji_starry_eyed or "50%" in body text can no longer break the whole compile."""
    body = _strip_fences(text)
    if "\\begin{document}" in body:
        body = body.split("\\begin{document}", 1)[1]
    body = body.replace("\\end{document}", "")
    # Unwrap a token list the model wrongly put in display math (else it renders as italic math):
    # \[ [a, b, c] \] -> [a, b, c]. Only when the content is a single bracketed list (starts '[', no '=').
    body = re.sub(r"\\\[\s*(\[[^\]=]*\])\s*\\\]", r"\1", body)
    # Drop model-drawn pipe "tables" (word | word | word) — they render as literal pipes, not a table;
    # the real table is the protected tabularx. A genuine LaTeX table uses '&', not '|', so this is safe.
    body = re.sub(r"(?m)^\s*[A-Za-z0-9'().\-]+(?: [A-Za-z0-9'().\-]+)*(?:\s*\|\s*[A-Za-z0-9'().\- ]+){2,}\s*$", "", body)

    out = []
    i, n = 0, len(body)
    inline = display = False        # $...$  and  $$...$$
    math_depth = 0                  # \[ \] and math environments
    table_depth = 0                 # tabular family
    while i < n:
        c = body[i]
        if c == "\\":
            two = body[i:i + 2]
            if two == "\\[":
                math_depth += 1; out.append(two); i += 2; continue
            if two == "\\]":
                if math_depth > 0:                       # drop a stray \] (no open \[) — it aborts the compile
                    math_depth -= 1; out.append(two)
                i += 2; continue
            m = _BE_RE.match(body, i)
            if m:
                env, opening = m.group(2), m.group(1) == "begin"
                if env in _MATH_ENVS:
                    math_depth = max(0, math_depth + (1 if opening else -1))
                elif env in _TABLE_ENVS:
                    table_depth = max(0, table_depth + (1 if opening else -1))
                out.append(m.group(0)); i = m.end(); continue
            out.append(body[i:i + 2]); i += 2; continue   # already-escaped char: copy verbatim
        if c == "$":
            mc = _CURRENCY.match(body, i)
            # "$1 billion" / "$37.42" is literal currency, but "$1$" is math — only escape when the
            # number is NOT immediately closed by another '$' (which would be a real inline-math span).
            if mc and (mc.end() >= n or body[mc.end()] != "$") and not (inline or display or math_depth):
                out.append("\\$" + mc.group(1)); i = mc.end(); continue
            if body[i:i + 2] == "$$":
                display = not display; out.append("$$"); i += 2; continue
            inline = not inline; out.append("$"); i += 1; continue
        in_math = inline or display or math_depth > 0
        if c == "&":
            out.append("&" if (in_math or table_depth > 0) else "\\&"); i += 1; continue
        if c in "_#%" and not in_math:
            out.append("\\" + c); i += 1; continue
        out.append(c); i += 1
    tail = "".join(out)
    if math_depth > 0:                                   # close any unclosed \[ so it can't truncate the doc
        tail += "\\]" * math_depth
    if inline:                                           # close a dangling inline-math $
        tail += "$"
    return _balance_envs(tail.strip())


def _balance_envs(body: str) -> str:
    """Append any missing \\end{...} for environments the model left open (e.g. an itemize inside a card),
    in LIFO order. Without this an unclosed list silently swallows the following content (and, since topic
    bodies are concatenated, the next topic too)."""
    stack = []
    for kind, env in _BE_RE.findall(body):
        if kind == "begin":
            stack.append(env)
        elif env in stack:
            for j in range(len(stack) - 1, -1, -1):
                if stack[j] == env:
                    del stack[j]
                    break
    if stack:
        body += "\n" + "\n".join("\\end{%s}" % e for e in reversed(stack))
    return body


# ---------------------------------------------------------------------------
# Staged pipeline: Content -> Review -> Structure. Worked examples/matrix/table
# are frozen as the opaque token [[EXAMPLES]] in stage 1 and carried untouched
# through every stage, so no stage can re-author, duplicate, or break them; the
# caller substitutes the real LaTeX only at the very end.
# ---------------------------------------------------------------------------

def _ollama(system: str, prompt: str, max_tokens: int = 1200, temperature: float = 0.3) -> str:
    payload = json.dumps({"model": OLLAMA_MODEL, "system": system, "prompt": prompt, "stream": False,
                          "options": {"temperature": temperature, "num_predict": max_tokens}}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()).get("response", "")


CONTENT_RULES = (
    "You pull the teachable CONTENT for ONE revision topic — nothing else. Output a flat numbered list of "
    "concise, self-contained points: the definition(s), the mechanism/how it works, key tested facts, and "
    "trade-offs. Ground every point in the LECTURE SLIDES; where the on-topic TEXTBOOK CONTEXT adds a named "
    "fact, algorithm, or error case, fold it into a point and state it accurately. Write NO LaTeX, NO callout "
    "boxes, NO section headings, and NO worked examples/numbers/tables/matrices — worked examples are inserted "
    "separately, so where one belongs put the literal token [[EXAMPLES]] as its own numbered line. Do not "
    "invent anything not in the slides or on-topic textbook context. Output only the numbered points."
)


def content_points(topic: str, source_material: str, transcript: str = "", avoid_topics: list = None,
                   has_examples: bool = True, max_tokens: int = 900) -> str:
    """Stage 1 — coverage + correctness as plain numbered points; examples frozen as [[EXAMPLES]]."""
    parts = [f'TOPIC: "{topic}"', source_material[:12000]]
    if transcript.strip():
        parts += ["LECTURE TRANSCRIPT (extra context):", transcript[:3000]]
    if avoid_topics:
        parts.append("Already covered in earlier topics — do NOT repeat: " + "; ".join(avoid_topics))
    if not has_examples:
        parts.append("(There are no separate worked examples for this topic, so do NOT emit [[EXAMPLES]].)")
    parts.append("List the content points now.")
    return _ollama(CONTENT_RULES, "\n\n".join(parts), max_tokens, 0.3).strip()


REVIEW_RULES = (
    "You are a strict fact-checker for revision notes. You are given the SOURCE (lecture slides + on-topic "
    "textbook) and a draft numbered list of POINTS. Return a corrected numbered list: KEEP points supported by "
    "the source; FIX any that misstate it; DELETE any unsupported, off-topic, or duplicated point. CHECK "
    "ESPECIALLY: (a) when two related terms are contrasted (e.g. stemming vs lemmatisation, precision vs "
    "recall), their definitions are NOT swapped — verify each against the source; (b) any specific NUMBER, "
    "RANGE, or figure (e.g. a similarity range, a duration, a count) actually appears in the source — DELETE "
    "invented ones; (c) a stated transform example matches the source (do not 'simplify' it to a no-op). Keep "
    "the literal token [[EXAMPLES]] if present. Add no new facts. Output only the corrected numbered points."
)


def review_points(topic: str, points: str, source_material: str, max_tokens: int = 900) -> str:
    """Stage 2 — verify each point against the source; fix/drop the odd ones (the reasoning slips)."""
    prompt = (f'TOPIC: "{topic}"\n\nSOURCE:\n{source_material[:12000]}\n\nDRAFT POINTS:\n{points}\n\n'
              "Return the corrected, source-supported numbered points.")
    return _ollama(REVIEW_RULES, prompt, max_tokens, 0.2).strip()


# Auto-generated gold-standard exemplars (Claude-written) teaching CONSOLIDATED structure: comparison ->
# one table, grouped cards, the [[EXAMPLES]] token kept verbatim and interpreted. Loaded from a file so the
# set can be regenerated; falls back to a minimal inline example if the file is absent.
def _load_structure_fewshot() -> str:
    try:
        return (Path(__file__).parent / "structure_fewshot.tex").read_text(encoding="utf-8").strip()
    except OSError:
        return ("POINTS:\n1. A document is one piece of text.\n2. [[EXAMPLES]]\n\nLATEX:\n\\section{Docs}\n"
                "\\begin{defbox}[Document]\nOne piece of text.\n\\end{defbox}\n[[EXAMPLES]]")


STRUCTURE_FEWSHOT = _load_structure_fewshot()

STRUCTURE_RULES = (
    "You are a LaTeX formatter for revision notes. Turn the reviewed POINTS into one styled \\section using "
    "ONLY these macros: \\section{...}, \\subsection{...}, \\begin{defbox}[Term]..\\end{defbox} (definition), "
    "\\begin{factbox}..\\end{factbox} (key fact), \\begin{card}[Title]..\\end{card} (summary itemize), "
    "\\begin{trapbox}[Label]..\\end{trapbox} (mistake), \\begin{pitfallbox}..\\end{pitfallbox}. "
    "HARD RULES: keep the token [[EXAMPLES]] EXACTLY as-is on its own line at TOP LEVEL — never edit, expand, "
    "delete, or place it inside any box — with one lead-in sentence before it and a one-sentence takeaway "
    "after it, both drawn from the points. Add NO new facts, numbers, examples, tables, or matrices; only "
    "arrange and style the points you are given. Never use '|' to draw a table. STRUCTURE — CONSOLIDATE, do "
    "NOT fragment: put a comparison of two OR MORE things in ONE tabularx table, never parallel per-item "
    "cards or a stack of one-line boxes; group several related facts into ONE card rather than many separate "
    "Fact boxes; aim for FEW substantive boxes per section. A card or box title must name its specific "
    "content ('Why k-NN predictions are slow'), never a generic label like 'Properties' or 'Key Facts', and "
    "must not repeat the heading above it. NEVER nest a box inside another. factbox and pitfallbox take NO "
    "label; trapbox/dobox labels have no leading colon. NOT everything needs a box — most explanation should "
    "be plain prose; reserve boxes for definitions, key facts, traps, and worked examples. Output ONLY the "
    "LaTeX body."
)


def _bracket_label(m):
    label = m.group(2).strip().rstrip(".:")
    if not label or len(label.split()) > 3 or "." in label:
        return m.group(0)               # a short same-line BODY sentence, not a leaked label: leave it
    return "%s[%s]" % (m.group(1), label[:1].upper() + label[1:])


def _hoist_inner(m):
    inner = _BLOCK_RE.search(m.group(3))
    if not inner or inner.group(1) == m.group(1):
        return m.group(0)
    content = m.group(3)[:inner.start()] + m.group(3)[inner.end():]
    return "\\begin{%s}%s%s\\end{%s}\n\n%s" % (m.group(1), m.group(2) or "", content, m.group(1), inner.group(0))


def _unnest_boxes(body: str) -> str:
    """Hoist a box nested inside a DIFFERENT box out to after the outer \\end (the model occasionally nests
    a worked-trace dobox inside a card; it renders box-in-box with the trace visually subordinated)."""
    prev = None
    while prev != body:
        prev = body
        body = _BLOCK_RE.sub(_hoist_inner, body)
    return body


def _fix_box_titles(body: str) -> str:
    """Repair callout-box glitches: factbox/pitfallbox take NO label (a stray one renders as 'Fact
    Punctuation'); strip a leading ': ' from any box label (renders as 'Worked trace : x'); bracket a
    short label the model left outside the brackets ('\\begin{trapbox}normalisation' leaks the word into
    the body text); and hoist a box nested inside a different box."""
    body = re.sub(r"(\\begin\{(?:factbox|pitfallbox)\})\s*\[[^\]]*\]", r"\1", body)
    body = re.sub(r"(\\begin\{(?:card|defbox|trapbox|dobox|quizbox)\})\[\s*:\s*", r"\1[", body)
    body = re.sub(r"(\\begin\{(?:card|defbox|trapbox|dobox|quizbox)\})[ \t]*:?[ \t]*"
                  r"([^\s\[\]\\{}][^\[\]\\\n{}]{0,50}?)[ \t]*$",
                  _bracket_label, body, flags=re.M)
    return _unnest_boxes(body)


def structure_points(topic: str, points: str, max_tokens: int = 1300) -> str:
    """Stage 3 — arrange reviewed points into styled LaTeX, keeping [[EXAMPLES]] untouched (sanitized)."""
    prompt = ("EXAMPLES OF THE TASK (separated by =====):\n%s\n\nNow format these POINTS for the topic "
              "\"%s\" the same way (keep [[EXAMPLES]] verbatim):\n\nPOINTS:\n%s\n\nLATEX:"
              % (STRUCTURE_FEWSHOT, topic, points))
    return _fix_box_titles(_sanitize_body(_ollama(STRUCTURE_RULES, prompt, max_tokens, 0.3)))


SCOPE_MODEL = os.getenv("SCOPE_MODEL", OLLAMA_MODEL)  # the verifier task is simple; can use a small model


def verify_relevant(topic: str, passages: list) -> list:
    """Scope-linking gate: keep only the passages that GENUINELY explain `topic`, judged by a local LLM,
    not just ones that share a surface word (which embedding cosine over-retrieves). Returns the filtered
    list; fail-open (returns the input) on any connection/parse error so grounding never silently vanishes."""
    if not passages:
        return passages
    listing = "\n".join("[%d] %s" % (i + 1, p[:400].replace("\n", " ")) for i, p in enumerate(passages))
    prompt = (
        'TOPIC: "%s" (a topic taught in an NLP text-preprocessing lecture).\n\n'
        "Numbered TEXTBOOK PASSAGES:\n%s\n\n"
        "Judge what each passage is PRIMARILY about. Reply with the numbers of ONLY the passages whose primary "
        "subject IS this exact topic and that would directly help teach it. REJECT a passage whose primary "
        "subject is a DIFFERENT NLP task — e.g. word-sense disambiguation, part-of-speech tagging, parsing, "
        "machine translation, dialogue systems, named-entity recognition — even when it shares a word such as "
        "'corpus', 'encoding', 'feature', or 'vector' with the topic. When unsure, REJECT. Reply with just the "
        "numbers separated by commas, or the single word NONE." % (topic, listing)
    )
    payload = json.dumps({"model": SCOPE_MODEL, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0, "num_predict": 40}}).encode()
    try:
        req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.loads(r.read()).get("response", "")
    except Exception:
        return passages  # fail-open: never lose grounding because the verifier was unreachable
    if "none" in resp.lower() and not re.search(r"\d", resp):
        return []
    keep = {int(n) for n in re.findall(r"\d+", resp) if 1 <= int(n) <= len(passages)}
    return [p for i, p in enumerate(passages, 1) if i in keep]


_BLOCK_RE = re.compile(
    r"\\begin\{(card|defbox|factbox|trapbox|pitfallbox|quizbox|dobox)\}(\[[^\]]*\])?(.*?)\\end\{\1\}",
    re.DOTALL,
)


_SECTION_RE = re.compile(r"\\section\*?\{([^}]*)\}")
_SUBSECTION_RE = re.compile(r"\\subsection\*?\{([^}]*)\}")


def _canon(title: str) -> str:
    """Canonical title for matching across level and spelling: lowercase + unify -ize/-ise so a
    'Tokenization' subsection matches a 'Tokenisation' section."""
    t = " ".join(title.split()).lower()
    return t.replace("ization", "isation").replace("ize", "ise")


def _dedup_subsections(content: str, drop: set = None, seen: set = None, own: str = None) -> str:
    """Drop \\subsections whose canonical title repeats — within this section (`seen`, shared across all
    sections) or that re-teach an existing \\section (`drop` = the set of section titles), e.g. a 'Review
    Analysis' section re-deriving 'Normalisation'/'Tokenisation'. Its own section title is exempt."""
    subs = list(_SUBSECTION_RE.finditer(content))
    if not subs:
        return content
    drop = drop or set()
    seen = seen if seen is not None else set()
    out = [content[:subs[0].start()]]
    for i, m in enumerate(subs):
        c = _canon(m.group(1))
        cw = set(c.split())
        end = subs[i + 1].start() if i + 1 < len(subs) else len(content)
        # also drop a subsection that RE-COVERS another section: its title contains all words of some
        # section title (e.g. "Text Normalization" re-covers the "Normalisation" section) — but never its own.
        re_covers = any(d and d != own and set(d.split()) <= cw for d in drop)
        if c in seen or (c in drop and c != own) or re_covers:
            continue
        seen.add(c)
        out.append(content[m.start():end])
    return "".join(out)


_CARD_RE = re.compile(
    r"\\begin\{(card|defbox|factbox|trapbox|pitfallbox|dobox)\}\[([^\]]*)\](.*?)\\end\{\1\}", re.DOTALL)


def _strip_recover_boxes(content: str, sections: set, own: str) -> str:
    """Drop a titled callout box that re-teaches a topic which already has its own \\section — e.g. a
    'Review Analysis' section carrying [Text Normalization]/[Tokenization] cards that re-cover §2/§3.
    (Subsection-level dedup misses these because the re-teaching is inside cards, not headings.)"""
    def keep(m):
        title = _canon(re.sub(r"\bcard\b", "", m.group(2)).lstrip(": ").strip())
        tw = set(title.split())
        if tw and any(d and d != own and set(d.split()) <= tw for d in sections):
            return ""
        return m.group(0)
    return _CARD_RE.sub(keep, content)


def dedup_sections(body: str) -> str:
    """Merge \\section blocks that share a title into one (first-seen order), then drop \\subsections AND
    callout boxes that repeat — within or across sections, or that re-teach a topic that already has its own
    \\section. Lossless for genuinely new content; no repeated heading, cross-level subsection, or re-teaching
    card survives."""
    secs = list(_SECTION_RE.finditer(body))
    if not secs:
        return _dedup_subsections(body)
    order, merged = [], {}
    for i, m in enumerate(secs):
        norm = _canon(m.group(1))
        end = secs[i + 1].start() if i + 1 < len(secs) else len(body)
        content = body[m.end():end]
        if norm not in merged:
            merged[norm] = [m.group(0), content]
            order.append(norm)
        else:
            merged[norm][1] += "\n" + content
    section_canons = set(order)
    seen_subs = set()
    out = [body[:secs[0].start()]]
    for norm in order:
        heading, content = merged[norm]
        content = _dedup_subsections(content, drop=section_canons, seen=seen_subs, own=norm)
        out.append(heading + _strip_recover_boxes(content, section_canons, norm))
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()


def _tidy_headings(body: str) -> str:
    """Drop an empty heading layer: a \\subsection whose entire content is ONE box (the heading wraps
    nothing else, adding navigation noise), or whose title is just echoed by the label of the box placed
    directly under it (the box's own title bar already says it)."""
    subs = list(_SUBSECTION_RE.finditer(body))
    if not subs:
        return body
    marks = sorted([m.start() for m in _SECTION_RE.finditer(body)]
                   + [m.start() for m in subs] + [len(body)])
    out, prev_end = [], 0
    for m in subs:
        nxt = min(p for p in marks if p > m.start())
        content = body[m.end():nxt]
        lone = re.fullmatch(
            r"\s*\\begin\{(card|defbox|factbox|trapbox|pitfallbox|quizbox|dobox)\}"
            r"(\[[^\]]*\])?.*?\\end\{\1\}\s*", content, re.DOTALL)
        first = _BLOCK_RE.search(content)
        echo = (first and first.group(2) and not content[:first.start()].strip()
                and _canon(first.group(2).strip("[] ")) == _canon(m.group(1)))
        if lone or echo:
            out.append(body[prev_end:m.start()])
            prev_end = m.end()                 # drop just the heading; the box stays
    out.append(body[prev_end:])
    return "".join(out)


def dedup_blocks(body: str) -> str:
    """Deterministically drop later exact-duplicate callout boxes (same environment, title, and inner
    text). The LLM merge pass routinely misses these across chunks, so a verbatim card can appear twice."""
    seen = set()

    def keep(m):
        key = (m.group(1), m.group(2) or "", " ".join(m.group(3).split()))
        if key in seen:
            return ""
        seen.add(key)
        return m.group(0)

    out = _BLOCK_RE.sub(keep, body)
    return re.sub(r"\n{3,}", "\n\n", out).strip()
