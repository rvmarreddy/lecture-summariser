import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from diagrams import inject_diagrams

_PANDOC = shutil.which("pandoc")
# xelatex has no glyphs for emoji / Notion decoration; strip them before compiling.
_DROP_GLYPHS = re.compile(r"[\U0001F000-\U0001FAFF☀-➿️]")


def timestamped_output(stem: str = "lecture_notes", out_dir: str = "outputs") -> str:
    """Return 'outputs/<stem>_<YYYYMMDD_HHMMSS>' (no extension); callers append .md/.pdf."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(out_dir) / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}")


def to_pdf(markdown: str, out_path: str = "lecture_notes.pdf", title: str = None) -> str:
    """Render detected diagrams into the Markdown, then compile to PDF via pandoc + xelatex."""
    if not _PANDOC:
        raise RuntimeError("pandoc not found on PATH")
    md = _DROP_GLYPHS.sub("", inject_diagrams(markdown))
    if title:
        md = f"# {title}\n\n{md}"
    # Distinct intermediate name so we never clobber a user-facing .md; kept in cwd so the
    # relative diagram image paths resolve. Removed after a successful compile.
    md_file = Path(out_path).with_name(Path(out_path).stem + ".pandoc.md")
    md_file.write_text(md, encoding="utf-8")
    subprocess.run(
        [_PANDOC, str(md_file), "-o", out_path, "--standalone",
         "--pdf-engine=xelatex", "-V", "geometry:margin=1in"],
        check=True, capture_output=True,
    )
    md_file.unlink(missing_ok=True)
    return out_path


def _tex_escape(s: str) -> str:
    """Escape plain-text fields (title/header/author) that get substituted into the LaTeX preamble."""
    repl = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
    return "".join(repl.get(c, c) for c in s)


def latex_to_pdf(body: str, out_path: str, title: str = "Lecture Notes", subtitle: str = "",
                 head_l: str = "", head_r: str = "", author: str = "") -> str:
    """Wrap a model-generated LaTeX body in the styled preamble template and compile via xelatex.

    `body` is LaTeX (left as-is); the title/header/author fields are plain text and get escaped.
    """
    template = (Path(__file__).parent / "note_preamble.tex").read_text(encoding="utf-8")
    body = _DROP_GLYPHS.sub("", body)  # xelatex/Helvetica Neue has no emoji glyphs -> tofu boxes; drop them
    doc = (template
           .replace("@@TITLE@@", _tex_escape(title or "Lecture Notes"))
           .replace("@@SUBTITLE@@", _tex_escape(subtitle))
           .replace("@@HEADL@@", _tex_escape(head_l))
           .replace("@@HEADR@@", _tex_escape(head_r or title))
           .replace("@@AUTHOR@@", _tex_escape(author))
           .replace("@@BODY@@", body))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    texf = out.with_suffix(".tex")
    texf.write_text(doc, encoding="utf-8")
    # Run twice so fancyhdr / section numbering settle. nonstopmode keeps going past minor
    # glitches in model-generated LaTeX so we still get a PDF.
    for _ in range(2):
        subprocess.run(["xelatex", "-interaction=nonstopmode", texf.name],
                       cwd=str(out.parent), capture_output=True)
    if not out.exists():
        raise RuntimeError(f"xelatex failed to produce {out}")
    return str(out)
