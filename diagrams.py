import re
import shutil
import subprocess
from pathlib import Path

_DOT = shutil.which("dot")
_ARROWS = ("→", "⟶", "➝", "⇒", "=>", "->")


def _clean_label(s: str) -> str:
    s = re.sub(r'[*_`#]+', '', s).strip(" -•\t")
    return s.replace('"', "'")[:40]


def _chain_from_line(line: str):
    """Return the node labels of an 'A -> B -> C' chain on this line, or None."""
    norm = line
    for a in _ARROWS:
        norm = norm.replace(a, "->")
    if norm.count("->") < 2:
        return None
    parts = [_clean_label(p) for p in norm.split("->")]
    parts = [p for p in parts if p]
    return parts if len(parts) >= 3 else None


def pipeline_to_dot(parts) -> str:
    nodes = "\n".join(f'  n{i} [label="{p}"];' for i, p in enumerate(parts))
    edges = "\n".join(f"  n{i} -> n{i + 1};" for i in range(len(parts) - 1))
    return (
        "digraph G {\n  rankdir=LR;\n  bgcolor=transparent;\n"
        '  node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname=Helvetica];\n'
        f"{nodes}\n{edges}\n}}"
    )


def render_dot(dot_src: str, out_path: str):
    """Render DOT to out_path (format taken from extension). Returns path, or None on failure."""
    if not _DOT:
        return None
    out = Path(out_path)
    try:
        subprocess.run(
            [_DOT, f"-T{out.suffix.lstrip('.')}", "-o", str(out)],
            input=dot_src.encode(), check=True, capture_output=True,
        )
        return str(out)
    except subprocess.CalledProcessError:
        return None


def inject_diagrams(markdown: str, out_dir: str = "diagrams_output") -> str:
    """Render detected pipelines to images and insert a reference after each source line."""
    Path(out_dir).mkdir(exist_ok=True)
    out_lines, idx = [], 0
    for line in markdown.splitlines():
        out_lines.append(line)
        parts = _chain_from_line(line)
        if parts:
            img = render_dot(pipeline_to_dot(parts), f"{out_dir}/diagram_{idx}.png")
            if img:
                out_lines += ["", f"![pipeline diagram]({img})", ""]
                idx += 1
    return "\n".join(out_lines)
