import re
import shutil
import subprocess
from pathlib import Path

from diagrams import inject_diagrams

_PANDOC = shutil.which("pandoc")
# xelatex has no glyphs for emoji / Notion decoration; strip them before compiling.
_DROP_GLYPHS = re.compile(r"[\U0001F000-\U0001FAFF☀-➿️]")


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
