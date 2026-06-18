import os
import re
import json
import urllib.request
from pathlib import Path

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    t = _THINK_RE.sub("", text).strip()   # drop reasoning blocks from thinking models (e.g. Qwen3)
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

def _ollama(system: str, prompt: str, max_tokens: int = 1200, temperature: float = 0.3,
            fmt: str = None) -> str:
    body = {"model": OLLAMA_MODEL, "system": system, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}}
    if fmt:                       # Ollama structured-output: "json" forces syntactically-valid JSON output
        body["format"] = fmt
    if "qwen3" in OLLAMA_MODEL.lower():
        body["think"] = False     # Qwen3 is a thinking model; disable so stages get the answer, not reasoning
    payload = json.dumps(body).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return _THINK_RE.sub("", json.loads(r.read()).get("response", ""))   # strip any leaked <think> block


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


# Gold-standard exemplars teaching CONSOLIDATED structure: comparison ->
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


def _structure_points_llm(topic: str, points: str, max_tokens: int = 1300) -> str:
    """Fallback Stage 3 — let the model emit styled LaTeX directly (used only when the structured-JSON
    path below returns nothing parseable). Kept because a deterministic renderer that produces NO body is
    worse than a slightly-messy LLM one."""
    prompt = ("EXAMPLES OF THE TASK (separated by =====):\n%s\n\nNow format these POINTS for the topic "
              "\"%s\" the same way (keep [[EXAMPLES]] verbatim):\n\nPOINTS:\n%s\n\nLATEX:"
              % (STRUCTURE_FEWSHOT, topic, points))
    return _fix_box_titles(_sanitize_body(_ollama(STRUCTURE_RULES, prompt, max_tokens, 0.3)))


# ---------------------------------------------------------------------------
# Deterministic structure (Stage 3, primary path). The model no longer writes
# LaTeX — it emits a STRUCTURED JSON description of the section (typed blocks),
# and Python renders it. So comparisons ALWAYS become real tables, every box is
# balanced and correctly labelled, nothing nests, and titles are never leaked
# into the body: structure can no longer be the limiting factor. The model only
# does the semantic grouping (which point is a definition / a comparison / a
# trap) — the thing it is actually okay at.
# ---------------------------------------------------------------------------

STRUCTURE_JSON_SYS = (
    "You convert reviewed revision-note POINTS for ONE topic into a STRUCTURED JSON object that a renderer "
    "turns into LaTeX. Output ONLY one JSON object, no prose, no LaTeX, no markdown fences. Schema:\n"
    '{"title":"<section title>","blocks":[ <block>, ... ]}\n'
    "Each block is exactly one of:\n"
    '  {"t":"prose","text":"<a short paragraph>"}\n'
    '  {"t":"def","term":"<term>","body":"<1-2 sentence definition>"}\n'
    '  {"t":"fact","body":"<one key tested fact>"}\n'
    '  {"t":"card","title":"<specific title>","items":["<point>","<point>"]}\n'
    '  {"t":"table","title":"<what is compared>","columns":["<A>","<B>"],"rows":[{"label":"<aspect>","cells":["<A val>","<B val>"]}]}\n'
    '  {"t":"trap","title":"<the mistake>","body":"<why it is wrong>"}\n'
    '  {"t":"pitfall","body":"<a common pitfall>"}\n'
    '  {"t":"examples"}\n'
    "HARD RULES: Put any comparison of two OR MORE things in ONE table block — never parallel cards or a "
    "stack of boxes. Group related facts into ONE card, not many fact blocks. Use def for definitions, fact "
    "for a single key fact, trap/pitfall for mistakes; everything else is prose. Most content should be "
    "prose — reserve boxes for definitions, key facts, traps and worked examples. Keep inline math in $...$. "
    "Invent NOTHING: every block must come from the given points. Each table row's cells array MUST have "
    'exactly len(columns) entries. If a point is the literal token [[EXAMPLES]], emit an {"t":"examples"} '
    "block at that position (at most once), with a one-sentence prose lead-in before it and a one-sentence "
    "prose takeaway after it, both drawn from the points."
)

# A worked input->JSON exemplar, built from a dict so the LaTeX backslashes need no hand-escaping.
_FEWSHOT_INPUT = (
    'TOPIC: "Precision vs Recall"\nPOINTS:\n'
    "1. Precision and recall are evaluation metrics for binary classifiers, from confusion-matrix counts TP, FP, FN.\n"
    "2. Precision = TP/(TP+FP): of items flagged positive, the fraction actually positive.\n"
    "3. Recall = TP/(TP+FN): of actually-positive items, the fraction retrieved.\n"
    "4. They trade off through the decision threshold.\n"
    "5. Precision is governed by false positives, recall by false negatives; neither uses true negatives.\n"
    "6. F1 = 2PR/(P+R) combines them into one number.\n"
    "7. [[EXAMPLES]]\n"
    "8. Predicting positive for everything gives 100% recall but useless precision."
)
_FEWSHOT_SECTION = {
    "title": "Precision vs Recall",
    "blocks": [
        {"t": "prose", "text": "Precision and recall are evaluation metrics for a binary classifier, both "
         "computed from confusion-matrix counts (TP, FP, FN). Neither uses true negatives, so both ignore "
         "the negative class."},
        {"t": "def", "term": "Precision", "body": "$\\text{Precision}=\\dfrac{TP}{TP+FP}$ — of all items "
         "flagged positive, the fraction actually positive. Governed by false positives."},
        {"t": "def", "term": "Recall", "body": "$\\text{Recall}=\\dfrac{TP}{TP+FN}$ — of all actually-"
         "positive items, the fraction retrieved. Governed by false negatives."},
        {"t": "table", "title": "Precision vs recall", "columns": ["Precision", "Recall"], "rows": [
            {"label": "Formula", "cells": ["$TP/(TP+FP)$", "$TP/(TP+FN)$"]},
            {"label": "Penalised by", "cells": ["False positives", "False negatives"]},
            {"label": "Optimise when", "cells": ["False alarms costly (spam)", "Misses costly (fraud)"]}]},
        {"t": "prose", "text": "When both error types matter, the F1 score $2PR/(P+R)$ combines them. The "
         "worked example below scores a classifier that labels everything positive."},
        {"t": "examples"},
        {"t": "prose", "text": "Labelling everything positive drives recall to 100% while precision falls to "
         "the base rate — the two must always be read together."},
        {"t": "trap", "title": "Precision is not accuracy", "body": "On an imbalanced dataset accuracy can "
         "look high while recall is near zero, because the majority negative class dominates."}]}
_STRUCT_FEWSHOT_JSON = ("EXAMPLE\n" + _FEWSHOT_INPUT + "\n\nJSON:\n"
                        + json.dumps(_FEWSHOT_SECTION, ensure_ascii=False))

# Strip section/environment macros the model must not put in a text field; inline $...$ math is left alone.
_STRIP_MACRO = re.compile(r"\\(?:sub)?section\*?\s*\{[^}]*\}|\\(?:begin|end)\s*\{[^}]*\}")


def _plain(s) -> str:
    """A renderer text field: drop any leaked structural macros and the [[EXAMPLES]] token (only the
    examples block may emit it), collapse whitespace. Specials are escaped later by _sanitize_body."""
    s = _STRIP_MACRO.sub("", str(s or ""))
    s = re.sub(r"\[\[\s*EXAMPLES\s*\]\]", "", s)
    return " ".join(s.split()).strip()


def _render_table(b: dict) -> str:
    cols = [_plain(c) for c in (b.get("columns") or [])]
    n0 = len(cols)
    if n0 < 2:
        return ""
    parsed = []                                               # (label, cells padded to n0), empty rows dropped
    for r in (b.get("rows") or []):
        if not isinstance(r, dict):
            continue
        cells = ([_plain(c) for c in (r.get("cells") or [])] + [""] * n0)[:n0]
        label = _plain(r.get("label", ""))
        if not label and not any(cells):
            continue
        parsed.append((label, cells))
    if not parsed:
        return ""
    keep = [j for j in range(n0) if any(c[j] for _, c in parsed)]  # prune columns with no data in any row
    if len(keep) < 2:
        return ""
    cols = [cols[j] for j in keep]
    head = " & " + " & ".join("\\textbf{%s}" % c for c in cols) + " \\\\"
    lines = ["%s & %s \\\\" % (lab, " & ".join(c[j] for j in keep)) for lab, c in parsed]
    return ("\\begin{center}\\begin{tabularx}{\\linewidth}{l%s}\n\\toprule\n%s\n\\midrule\n%s\n\\bottomrule\n"
            "\\end{tabularx}\\end{center}" % ("X" * len(cols), head, "\n".join(lines)))


def _render_section(section: dict) -> str:
    """Deterministically render a parsed section dict to clean LaTeX. Pure function of the JSON — it cannot
    leak a label, nest a box, leave an environment open, or flatten a comparison into a box stack."""
    if not isinstance(section, dict):
        return ""
    parts = ["\\section{%s}" % (_plain(section.get("title")) or "Notes")]
    for b in section.get("blocks", []):
        if not isinstance(b, dict):
            continue
        t = (b.get("t") or "").lower()
        if t == "prose":
            x = _plain(b.get("text", ""))
            if x:
                parts.append(x)
        elif t == "def":
            body = _plain(b.get("body", ""))
            if body:
                parts.append("\\begin{defbox}[%s]\n%s\n\\end{defbox}" % (_plain(b.get("term", "")), body))
        elif t == "fact":
            body = _plain(b.get("body", ""))
            if body:
                parts.append("\\begin{factbox}\n%s\n\\end{factbox}" % body)
        elif t == "card":
            items = [_plain(i) for i in (b.get("items") or [])]
            items = [i for i in items if i]
            if items:
                lst = "\n".join("  \\item %s" % i for i in items)
                parts.append("\\begin{card}[%s]\n\\begin{itemize}\n%s\n\\end{itemize}\n\\end{card}"
                             % (_plain(b.get("title", "")), lst))
        elif t == "table":
            tbl = _render_table(b)
            if tbl:
                parts.append(tbl)
        elif t == "trap":
            body = _plain(b.get("body", ""))
            if body:
                parts.append("\\begin{trapbox}[%s]\n%s\n\\end{trapbox}" % (_plain(b.get("title", "")), body))
        elif t == "pitfall":
            body = _plain(b.get("body", ""))
            if body:
                parts.append("\\begin{pitfallbox}\n%s\n\\end{pitfallbox}" % body)
        elif t == "examples":
            parts.append("[[EXAMPLES]]")
    return "\n\n".join(parts) if len(parts) > 1 else ""


def _parse_section(raw: str):
    """Robustly parse the model's JSON section (tolerates fences / leading prose) — returns a dict with a
    'blocks' list, or None so the caller can fall back to the LLM-LaTeX path."""
    if not raw or not raw.strip():
        return None
    t = _strip_fences(raw).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(t[start:end + 1])
    except Exception:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("blocks"), list) and obj["blocks"]:
        return obj
    return None


def structure_blocks(topic: str, points: str, max_tokens: int = 1800):
    """Stage 3 primary — ask the model for a STRUCTURED JSON section (json-mode) and parse it."""
    prompt = ("%s\n\nNow do the same for this topic. Output ONLY the JSON object.\nTOPIC: \"%s\"\nPOINTS:\n%s\n\nJSON:"
              % (_STRUCT_FEWSHOT_JSON, topic, points))
    raw = _ollama(STRUCTURE_JSON_SYS, prompt, max_tokens, 0.2, fmt="json")
    return _parse_section(raw)


# STRUCTURE_MODE=llm forces the old free-LaTeX path (used for A/B comparison); default is the deterministic
# JSON renderer.
STRUCTURE_MODE = os.getenv("STRUCTURE_MODE", "json").lower()


def structure_points(topic: str, points: str, max_tokens: int = 1800) -> str:
    """Stage 3 — arrange reviewed points into styled LaTeX. Primary path is the deterministic renderer over a
    structured-JSON section; falls back to direct LLM LaTeX only if the JSON is unusable. [[EXAMPLES]] is kept
    verbatim for the caller to substitute."""
    if STRUCTURE_MODE == "llm":
        return _structure_points_llm(topic, points, max_tokens=1300)
    try:
        section = structure_blocks(topic, points, max_tokens)
    except Exception:
        section = None
    if section:
        rendered = _render_section(section)
        if rendered.strip():
            return _fix_box_titles(_sanitize_body(rendered))
    return _structure_points_llm(topic, points, max_tokens=1300)


SCOPE_MODEL = os.getenv("SCOPE_MODEL", OLLAMA_MODEL)  # the verifier task is simple; can use a small model


def verify_relevant(topic: str, passages: list, other_topics: list = None) -> list:
    """Scope-linking gate: keep only the passages that GENUINELY explain `topic`, judged by a local LLM,
    not just ones that share a surface word (which embedding cosine over-retrieves). The "different subject"
    guidance is derived from the deck's OWN other topics (`other_topics`) rather than a hard-coded NLP-task
    list, so the gate generalises to any course. Returns the filtered list; fail-open (returns the input) on
    any connection/parse error so grounding never silently vanishes."""
    if not passages:
        return passages
    listing = "\n".join("[%d] %s" % (i + 1, p[:400].replace("\n", " ")) for i, p in enumerate(passages))
    tl = topic.strip().lower()
    others = [t for t in (other_topics or []) if t and t.strip().lower() != tl][:12]
    other_clause = ((" A DIFFERENT subject here means another topic from this same course, for example: %s."
                     % "; ".join(others)) if others else "")
    prompt = (
        'TOPIC: "%s".\n\n'
        "Numbered TEXTBOOK PASSAGES:\n%s\n\n"
        "Judge what each passage is PRIMARILY about. Reply with the numbers of ONLY the passages whose primary "
        "subject IS this exact topic and that would directly help teach it. REJECT a passage whose primary "
        "subject is a DIFFERENT subject, even when it shares a surface word (such as 'corpus', 'encoding', "
        "'feature', 'vector', or 'model') with the topic.%s When unsure, REJECT. Reply with just the numbers "
        "separated by commas, or the single word NONE." % (topic, listing, other_clause)
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


# ---------------------------------------------------------------------------
# Independent NLI fact-check. The Review stage is the SAME generative model that
# wrote the points grading itself, so it shares the writer's blind spots. This
# adds a genuinely independent signal: a cross-encoder NLI model (a different
# architecture and failure-mode) scores each point's entailment against the
# source, and a point that the source actively CONTRADICTS is dropped — the
# swapped-definition / wrong-number class of error. Neutral (un-entailed but not
# contradicted) points are KEPT, so legitimate textbook enrichment is not nuked.
# Fail-open: if the model can't load (offline / not downloaded), points pass
# through unchanged. Upgrade path: a dedicated checker such as MiniCheck.
# ---------------------------------------------------------------------------

NLI_MODEL = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
NLI_ENABLE = os.getenv("NLI_ENABLE", "1").lower() not in ("0", "false", "no", "")
# cross-encoder/nli-* logits are ordered [contradiction, entailment, neutral].
NLI_ENTAIL_MIN = float(os.getenv("NLI_ENTAIL_MIN", "0.15"))   # below this entailment AND...
NLI_CONTRA_MIN = float(os.getenv("NLI_CONTRA_MIN", "0.55"))   # ...above this contradiction -> drop
_nli_model = "uninitialised"


def _get_nli():
    global _nli_model
    if _nli_model == "uninitialised":
        try:
            from sentence_transformers import CrossEncoder
            _nli_model = CrossEncoder(NLI_MODEL)
        except Exception:
            _nli_model = None     # unavailable -> fail-open for the rest of the run
    return _nli_model


def _split_points(points: str) -> list:
    """Split a numbered list back into the individual point strings (preserving multi-line points)."""
    items, cur = [], None
    for line in points.splitlines():
        m = re.match(r"\s*\d+[.)]\s+(.*)", line)
        if m:
            if cur is not None:
                items.append(cur.strip())
            cur = m.group(1)
        elif cur is not None and line.strip():
            cur += " " + line.strip()
    if cur is not None:
        items.append(cur.strip())
    return [i for i in items if i]


def _softmax_rows(x):
    import numpy as np
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def nli_filter(points: str, source_material: str, keep_token: str = "[[EXAMPLES]]"):
    """Drop points the SOURCE contradicts, judged by an independent NLI cross-encoder. Returns
    (filtered_points, stats). Conservative: only removes clearly-contradicted points; keeps neutral ones."""
    if not NLI_ENABLE:
        return points, {"checked": 0, "dropped": 0}
    items = _split_points(points)
    if not items:
        return points, {"checked": 0, "dropped": 0}
    model = _get_nli()
    if model is None:
        return points, {"checked": 0, "dropped": 0, "error": "nli unavailable"}
    src = source_material.strip()
    chunks = [src[i:i + 1200] for i in range(0, len(src), 1200)] or [src]
    kept, dropped = [], 0
    for it in items:
        if keep_token in it:                          # never touch the frozen-examples placeholder
            kept.append(it)
            continue
        try:
            probs = _softmax_rows(model.predict([(c, it) for c in chunks]))   # premise=source, hyp=point
            ent = float(probs[:, 1].max())
            con = float(probs[:, 0].max())
        except Exception:
            kept.append(it)                           # scoring error -> keep (fail-open per point)
            continue
        if ent < NLI_ENTAIL_MIN and con >= NLI_CONTRA_MIN:
            dropped += 1
            continue
        kept.append(it)
    out = "\n".join("%d. %s" % (i + 1, t) for i, t in enumerate(kept))
    return (out or points), {"checked": len(items), "dropped": dropped}
