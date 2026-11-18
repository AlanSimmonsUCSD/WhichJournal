"""
textextract.py — pull the title and abstract out of an uploaded manuscript (PDF, DOCX, DOC).
Heuristic, so the result goes into the text box for the user to check and edit.
"""
from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile


def extract_text(data: bytes, filename: str, max_pages: int = 4) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages[:max_pages])
    if name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    if name.endswith(".doc"):
        # Old binary Word format: needs the `antiword` program (listed in packages.txt for
        # Streamlit Cloud; `brew install antiword` / `apt install antiword` locally).
        if not shutil.which("antiword"):
            raise RuntimeError("Reading .doc files needs 'antiword' installed — or re-save the file "
                               "as .docx or PDF.")
        with tempfile.NamedTemporaryFile(suffix=".doc") as f:
            f.write(data)
            f.flush()
            out = subprocess.run(["antiword", f.name], capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or "antiword could not read this file.")
        return out.stdout
    raise RuntimeError("Unsupported file type — use PDF, DOCX or DOC.")


_SKIP_TITLE = re.compile(
    r"^(running (head|title)|short title|title page|manuscript|original (article|research|paper)|"
    r"research (article|paper)|article type|word count|corresponding|keywords?|abstract|"
    r"submitted|received|accepted|journal|doi|page \d|\d+$)", re.I)
_ABSTRACT_HEAD = re.compile(r"(?im)^[ \t]*(abstract|summary)\b[ \t]*[:.\-—]?[ \t]*")
_ABSTRACT_END = re.compile(
    r"(?im)^[ \t]*(key[ \t]?words?|index terms|introduction|1\.?[ \t]+introduction|i\.[ \t]+introduction|"
    r"abbreviations|highlights|significance( statement)?|main text|plain language summary|"
    r"what is (already )?known|trial registration\b(?!.*:))\b")


def _clean(t: str) -> str:
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)       # re-join hyphenated line breaks
    t = re.sub(r"[ \t]*\n[ \t]*", " ", t)         # unwrap lines
    return re.sub(r"\s{2,}", " ", t).strip()


def find_title_abstract(text: str, max_abstract_chars: int = 4000) -> tuple[str, str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = ""
    for l in lines[:40]:
        l2 = re.sub(r"^title\s*[:\-]\s*", "", l, flags=re.I)
        words = len(l2.split())
        if 4 <= words <= 40 and not _SKIP_TITLE.match(l2) and "@" not in l2 and not l2.endswith(","):
            title = l2
            break
    m = _ABSTRACT_HEAD.search(text)
    if m:
        rest = text[m.end():]
        end = _ABSTRACT_END.search(rest)
        abstract = rest[: end.start()] if end else rest[:max_abstract_chars]
    else:  # no "Abstract" heading: take the text after the title
        start = text.find(title) + len(title) if title else 0
        abstract = text[start: start + 2500]
    return _clean(title), _clean(abstract)[:max_abstract_chars]
