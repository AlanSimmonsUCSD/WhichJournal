"""
authors.py — "how often has each suggested reviewer cited me?" using OpenAlex.

Steps
  1. Resolve each entered author (ORCID, or a name -> candidate list the user confirms) to an
     OpenAlex author ID.
  2. Fetch the entered authors' works (IDs, years, co-authors).
  3. Resolve each suggested reviewer to an OpenAlex author ID via the PubMed papers that made them
     a suggestion (PMID -> OpenAlex work -> the authorship whose name matches).
  4. Fetch every work by those reviewers that cites any of your works (deduplicated, exact count),
     plus each reviewer's total number of works (for a citation *rate*).
  5. Flag reviewers who have co-authored with you.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
import requests

ORCID_RE = re.compile(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dXx])")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", s.lower()).strip()


def _short(oa_id: str) -> str:
    return (oa_id or "").rsplit("/", 1)[-1]


def parse_entries(text: str) -> list[dict]:
    """Split 'Carpenter JR; 0000-0002-1825-0097, Jane Doe' into entries."""
    out = []
    for raw in re.split(r"[;\n]+", text or ""):
        raw = raw.strip()
        if not raw:
            continue
        m = ORCID_RE.search(raw)
        if m:
            out.append({"kind": "orcid", "value": m.group(1).upper(), "raw": raw})
        else:
            out.append({"kind": "name", "value": raw, "raw": raw})
    return out


def name_query(name: str) -> str:
    """PubMed style 'Carpenter JR' -> 'J R Carpenter' (OpenAlex search works better that way)."""
    toks = name.replace(",", " ").split()
    if len(toks) >= 2 and re.fullmatch(r"[A-Z]{1,3}", toks[-1]):
        return " ".join(list(toks[-1]) + toks[:-1])
    return " ".join(toks)


def _author_summary(a: dict) -> dict:
    inst = (a.get("last_known_institutions") or [{}])
    inst = inst[0].get("display_name", "") if inst else ""
    return {"id": _short(a.get("id", "")), "name": a.get("display_name", ""),
            "works": a.get("works_count", 0), "institution": inst or "",
            "orcid": (a.get("orcid") or "").rsplit("/", 1)[-1]}


def resolve_orcid(client, orcid: str) -> dict | None:
    try:
        return _author_summary(client.openalex(f"authors/orcid:{orcid}", {}))
    except requests.HTTPError:
        return None


def name_candidates(client, name: str, n: int = 6) -> list[dict]:
    data = client.openalex("authors", {"search": name_query(name), "per_page": n})
    return [_author_summary(a) for a in data.get("results", [])]


def _paged(client, path: str, params: dict, max_pages: int = 10) -> list[dict]:
    out, cursor = [], "*"
    for _ in range(max_pages):
        data = client.openalex(path, {**params, "per_page": 200, "cursor": cursor})
        out += data.get("results", [])
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
    return out


def my_works(client, author_ids: list[str]) -> list[dict]:
    """Works of the entered authors: id, year, author ids on each (for co-authorship)."""
    works = _paged(client, "works", {"filter": "authorships.author.id:" + "|".join(author_ids),
                                     "select": "id,publication_year,authorships"})
    return [{"id": _short(w["id"]), "year": w.get("publication_year"),
             "authors": {_short((a.get("author") or {}).get("id", "")) for a in w.get("authorships") or []}}
            for w in works]


def _filter_pmid(client, pmids: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(pmids), 50):
        chunk = "|".join(pmids[i:i + 50])
        params = {"select": "ids,authorships", "per_page": 200}
        try:
            data = client.openalex("works", {**params, "filter": f"pmid:{chunk}"})
        except requests.HTTPError:  # older/alternate filter name
            data = client.openalex("works", {**params, "filter": f"ids.pmid:{chunk}"})
        out += data.get("results", [])
    return out


def resolve_reviewers(client, reviewers: pd.DataFrame) -> dict[int, str]:
    """{reviewer row index: OpenAlex author id} using the PMIDs of their similar papers."""
    pmids = sorted({p for ps in reviewers["pmids"] for p in ps})
    works = _filter_pmid(client, pmids)
    by_pmid = {}
    for w in works:
        pm = ((w.get("ids") or {}).get("pmid") or "").rsplit("/", 1)[-1]
        if pm:
            by_pmid[pm] = w.get("authorships") or []
    found = {}
    for idx, r in reviewers.iterrows():
        last, init = _norm(r["last"]), _norm(r["initials"])[:1]
        votes: dict[str, int] = {}
        for pm in r["pmids"]:
            for au in by_pmid.get(pm, []):
                names = {_norm((au.get("author") or {}).get("display_name", "")),
                         _norm(au.get("raw_author_name", ""))}
                for nm in names:
                    toks = nm.split()
                    if toks and last and last.split()[-1] in toks and (not init or toks[0][:1] == init):
                        aid = _short((au.get("author") or {}).get("id", ""))
                        if aid:
                            votes[aid] = votes.get(aid, 0) + 1
                        break
        if votes:
            found[idx] = max(votes, key=votes.get)
    return found


def reviewer_work_counts(client, ids: list[str]) -> dict[str, int]:
    out = {}
    for i in range(0, len(ids), 50):
        chunk = "|".join(ids[i:i + 50])
        params = {"select": "id,works_count", "per_page": 200}
        try:
            data = client.openalex("authors", {**params, "filter": f"openalex:{chunk}"})
        except requests.HTTPError:
            data = client.openalex("authors", {**params, "filter": f"ids.openalex:{chunk}"})
        for a in data.get("results", []):
            out[_short(a["id"])] = a.get("works_count") or 0
    return out


def citing_counts(client, my_work_ids: list[str], reviewer_ids: list[str]) -> dict[str, set]:
    """{reviewer id: set of their work ids that cite any of my works} — exact, deduplicated."""
    hits: dict[str, set] = {r: set() for r in reviewer_ids}
    rset = set(reviewer_ids)
    for i in range(0, len(my_work_ids), 50):
        cites = "|".join(my_work_ids[i:i + 50])
        for k in range(0, len(reviewer_ids), 50):
            revs = "|".join(reviewer_ids[k:k + 50])
            works = _paged(client, "works", {"filter": f"cites:{cites},authorships.author.id:{revs}",
                                             "select": "id,authorships"}, max_pages=5)
            for w in works:
                wid = _short(w["id"])
                for a in w.get("authorships") or []:
                    aid = _short((a.get("author") or {}).get("id", ""))
                    if aid in rset:
                        hits[aid].add(wid)
    return hits


def citation_check(client, my_ids: list[str], reviewers: pd.DataFrame, this_year: int,
                   coauthor_years: int = 5) -> pd.DataFrame:
    """Columns to add to the reviewer table, indexed like `reviewers`."""
    mine = my_works(client, my_ids)
    my_work_ids = [w["id"] for w in mine]
    rmap = resolve_reviewers(client, reviewers)
    rids = sorted(set(rmap.values()) - set(my_ids))
    counts = reviewer_work_counts(client, rids) if rids else {}
    hits = citing_counts(client, my_work_ids, rids) if (rids and my_work_ids) else {}
    rows = {}
    for idx in reviewers.index:
        rid = rmap.get(idx)
        co = [w["year"] for w in mine if rid and rid in w["authors"]]
        n_cite = len(hits.get(rid, ())) if rid else np.nan
        total = counts.get(rid) if rid else None
        rows[idx] = {
            "openalex_author": rid or "",
            "Is you": bool(rid and rid in my_ids),
            "Cites you (papers)": n_cite,
            "Cite rate %": (100 * n_cite / total) if (rid and total) else np.nan,
            "Co-authored (papers)": len(co) if rid else np.nan,
            "Last co-authored": max((y for y in co if y), default=None),
            "Recent co-author": bool(co) and max((y or 0) for y in co) >= this_year - coauthor_years,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.attrs["n_my_works"] = len(my_work_ids)
    out.attrs["n_resolved"] = len(rmap)
    return out


def composite(score: pd.Series, cite_rate: pd.Series, weight: float = 0.5) -> pd.Series:
    """Weighted geometric mean of reviewer fit score and citation rate (both scaled 0-1, citation
    rate floored at 0.05 so non-citers are penalised, not zeroed). Rescaled so the best = 100."""
    fit = (score / 100).clip(lower=0.01)
    rate = cite_rate.fillna(0)
    rate_n = (rate / rate.max() if rate.max() > 0 else rate * 0).clip(lower=0) * 0.95 + 0.05
    c = fit ** (1 - weight) * rate_n ** weight
    return 100 * c / c.max()
