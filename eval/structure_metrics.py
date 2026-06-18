"""Objective (non-LLM) structural metrics for generated note .tex files. Quantifies whether the
deterministic renderer made structure clean: counts sections/tables/boxes and flags defect signatures
(pipe-tables, leaked box labels, nested boxes, unbalanced environments) + whether the PDF compiled.

Usage:  .venv/bin/python eval/structure_metrics.py eval/out/old_L3.tex eval/out/new_L3.tex ...
"""
import re
import sys
from pathlib import Path

BOXES = ["card", "defbox", "factbox", "trapbox", "pitfallbox", "quizbox", "dobox"]
_BE = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")
_BLOCK = re.compile(r"\\begin\{(%s)\}(\[[^\]]*\])?(.*?)\\end\{\1\}" % "|".join(BOXES), re.DOTALL)
# A box-begin immediately followed by body text instead of [ or newline = a leaked label.
_LEAK = re.compile(r"\\begin\{(?:%s)\}[ \t]*[^\[\s%%]" % "|".join(BOXES))
_PIPE = re.compile(r"(?m)^\s*[A-Za-z0-9'().\-]+(?: [A-Za-z0-9'().\-]+)*(?:\s*\|\s*[A-Za-z0-9'().\- ]+){2,}\s*$")


def body_of(tex: str) -> str:
    b = tex.split("\\begin{document}", 1)[-1] if "\\begin{document}" in tex else tex
    return b.split("\\end{document}", 1)[0]      # drop the lone \end{document} so it doesn't skew balance


def nested_boxes(body: str) -> int:
    n = 0
    for m in _BLOCK.finditer(body):
        inner = m.group(3)
        if re.search(r"\\begin\{(?:%s)\}" % "|".join(BOXES), inner):
            n += 1
    return n


def env_balance(body: str):
    stack, bad = [], 0
    for kind, env in _BE.findall(body):
        if kind == "begin":
            stack.append(env)
        else:
            if stack and stack[-1] == env:
                stack.pop()
            elif env in stack:
                stack.remove(env); bad += 1
            else:
                bad += 1
    return len(stack) + bad           # 0 == perfectly balanced


def metrics(path: Path) -> dict:
    tex = path.read_text(encoding="utf-8", errors="ignore")
    body = body_of(tex)
    pdf = path.with_suffix(".pdf")
    return {
        "file": path.name,
        "sections": len(re.findall(r"\\section\*?\{", body)),
        "subsections": len(re.findall(r"\\subsection\*?\{", body)),
        "tables": len(re.findall(r"\\begin\{tabularx?\}", body)),
        "boxes": sum(len(re.findall(r"\\begin\{%s\}" % b, body)) for b in BOXES),
        "DEFECT_pipe_tables": len(_PIPE.findall(body)),
        "DEFECT_leaked_labels": len(_LEAK.findall(body)),
        "DEFECT_nested_boxes": nested_boxes(body),
        "DEFECT_env_imbalance": env_balance(body),
        "pdf_ok": pdf.exists(),
        "body_chars": len(body.strip()),
    }


def main():
    rows = [metrics(Path(p)) for p in sys.argv[1:] if Path(p).exists()]
    if not rows:
        print("no .tex files found"); return
    cols = ["file", "sections", "subsections", "tables", "boxes", "DEFECT_pipe_tables",
            "DEFECT_leaked_labels", "DEFECT_nested_boxes", "DEFECT_env_imbalance", "pdf_ok", "body_chars"]
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(w[c]) for c in cols))
    print("-+-".join("-" * w[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(w[c]) for c in cols))


if __name__ == "__main__":
    main()
