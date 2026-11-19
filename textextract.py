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


# ---------------------------------------------------------------------------------------------
# Authors: the lines between the title and the abstract
# ---------------------------------------------------------------------------------------------
_DEGREES = {"md", "phd", "msc", "ms", "mph", "bsc", "ba", "bs", "ma", "mba", "rn", "dphil", "mbbs",
            "frcp", "frcpc", "facp", "do", "pharmd", "psyd", "scd", "drph", "mbchb", "frcpsych",
            "prof", "dr", "jr", "sr", "ii", "iii", "mrcpsych", "dsc", "bm", "bch", "mrcp", "faha"}
_NOT_NAMES = re.compile(
    r"(department|dept\.?|university|universit|institute|college|hospital|school|faculty|centre|"
    r"center|clinic|laboratory|division|unit\b|foundation|corresponding|correspondence|e-?mail|@|"
    r"address|tel\.?|phone|fax|street|road|avenue|usa|united|kingdom|running|word count|keywords|"
    r"orcid|equal contribution|contributed equally|received|accepted|abstract|summary)", re.I)


def _clean_name(raw: str) -> str:
    s = re.sub(r"[\d\*†‡§¶#∗,]+$", "", raw.strip())           # trailing affiliation markers
    s = re.sub(r"(?<=[A-Za-z\.])[\d\*†‡§¶#∗]+", "", s)          # markers glued to the name
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)                        # (ORCID ...) etc.
    toks = s.replace(",", " ").split()
    # "Carpenter JR" (PubMed style: surname + initials) — don't mistake initials for degrees/suffixes
    if len(toks) == 2 and re.fullmatch(r"[A-Z]{1,3}", toks[1]) and len(toks[0]) > 1:
        return " ".join(toks)
    toks = [t for t in toks if t.strip(".").lower() not in _DEGREES]
    return " ".join(toks).strip(" .;:-")


def _looks_like_name(s: str) -> bool:
    toks = s.split()
    if not 2 <= len(toks) <= 5 or _NOT_NAMES.search(s) or any(ch.isdigit() for ch in s):
        return False
    # every token is Capitalised, an initial ("J." / "JR"), or a particle (van, de, ...)
    particles = {"van", "von", "de", "der", "den", "da", "di", "du", "del", "la", "le", "ter", "bin", "al"}
    return all(t[0].isupper() or t.lower() in particles or t.startswith(("d'", "O'")) for t in toks)


def find_authors(text: str, title: str = "", max_authors: int = 30) -> list[str]:
    """Author names from the block between the title and the abstract (heuristic)."""
    head = text
    if title and title[:40] in text:
        head = text[text.index(title[:40]) + len(title[:40]):]
        head = head[head.find("\n") + 1:] if "\n" in head else head
    m = _ABSTRACT_HEAD.search(head)
    head = head[: m.start()] if m else head[:1500]
    names: list[str] = []
    for line in head.splitlines():
        line = line.strip()
        if not line or _NOT_NAMES.search(line) and not re.search(r",.*,", line):
            continue
        if re.match(r"^\s*[\d\*†‡§¶a-z]\s", line):  # "1 Department of ..." affiliation lines
            continue
        # split on commas / semicolons / " and " / "&", but keep "Smith, MD" together by
        # cleaning degrees after splitting
        for part in re.split(r"\s*(?:;|,|\band\b|&)\s*", line):
            nm = _clean_name(part)
            if _looks_like_name(nm) and nm not in names:
                names.append(nm)
    return names[:max_authors]
