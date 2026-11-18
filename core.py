"""
core.py — JANE-style journal recommender (no UI code here, so it can be reused/tested).

Pipeline
  1. Build a PubMed query from your abstract's keywords + domain (MeSH) + article-type filters.
  2. Fetch the top-N "Best Match" PubMed records (title, abstract, journal, ISSN).
  3. TF-IDF cosine similarity between your abstract and every candidate abstract.
  4. Keep the K most similar articles; a journal's FIT = sum of its articles' similarities
     (scaled 0-100), like JANE's confidence score.
  5. Look the journals up in OpenAlex (free, CC0) for impact (2-yr mean citedness ~ JIF,
     h-index), output volume, open-access status and APC.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENALEX = "https://api.openalex.org"

# --------------------------------------------------------------------------------------
# Domain filters (MeSH). Edit freely — these are deliberately broad.
# --------------------------------------------------------------------------------------
DOMAINS: dict[str, str] = {
    "Cardiology": '"Cardiovascular Diseases"[mh] OR "Cardiology"[mh] OR "Heart"[mh]',
    "Cell & Molecular Neuroscience": '"Neurons"[mh] OR "Neurosciences"[mh] OR '
    '"Synaptic Transmission"[mh] OR "Neuroglia"[mh] OR "Nervous System Physiological Phenomena"[mh]',
    "Endocrinology & Diabetes": '"Endocrine System Diseases"[mh] OR "Diabetes Mellitus"[mh] OR '
    '"Endocrinology"[mh] OR "Hormones"[mh]',
    "Immunology": '"Immune System Phenomena"[mh] OR "Immune System Diseases"[mh] OR '
    '"Allergy and Immunology"[mh]',
    "Infectious Diseases": '"Infections"[mh] OR "Communicable Diseases"[mh] OR "Anti-Infective Agents"[mh]',
    "Neurology": '"Nervous System Diseases"[mh] OR "Neurology"[mh]',
    "Oncology": '"Neoplasms"[mh] OR "Medical Oncology"[mh] OR "Antineoplastic Agents"[mh]',
    "Psychiatry": '"Mental Disorders"[mh] OR "Psychiatry"[mh] OR "Psychotropic Drugs"[mh] OR '
    '"Mental Health"[mh]',
}

# Used for the "also include methods/biostatistics journals" switch
METHODS_MESH = (
    '"Statistics as Topic"[mh] OR "Models, Statistical"[mh] OR "Research Design"[mh] OR '
    '"Randomized Controlled Trials as Topic"[mh] OR "Data Interpretation, Statistical"[mh]'
)

_NOT_SECONDARY = (
    'review[pt] OR "systematic review"[pt] OR "meta-analysis"[pt] OR editorial[pt] OR '
    'letter[pt] OR comment[pt] OR news[pt] OR "published erratum"[pt] OR "practice guideline"[pt]'
)

# --------------------------------------------------------------------------------------
# Article-type toggles (PubMed publication types). Selected types are OR-ed together.
# --------------------------------------------------------------------------------------
ARTICLE_TYPES: dict[str, str] = {
    "Novel data (original research)": f'"journal article"[pt] NOT ({_NOT_SECONDARY})',
    "Randomized controlled trial": '"randomized controlled trial"[pt]',
    "Clinical trial (any phase)": '"clinical trial"[pt]',
    "Observational study": '"observational study"[pt]',
    "Statistical / methods paper": METHODS_MESH,
    "Narrative review": 'review[pt] NOT "systematic review"[pt] NOT "meta-analysis"[pt]',
    "Systematic review": '"systematic review"[pt]',
    "Meta-analysis": '"meta-analysis"[pt]',
    "Trial protocol": '"clinical trial protocol"[pt]',
}

_EXTRA_STOP = {
    "study", "studies", "result", "results", "method", "methods", "background", "objective",
    "objectives", "conclusion", "conclusions", "aim", "aims", "using", "used", "use", "based",
    "however", "also", "among", "associated", "association", "found", "show", "showed", "shown",
    "significant", "significantly", "analysis", "analyses", "data", "paper", "propose", "proposed",
    "approach", "present", "new", "novel", "including", "include", "compared", "well", "within",
    "across", "provide", "provides", "important", "potential", "patients", "participants",
    "years", "year", "high", "higher", "lower", "increase", "increased", "may", "can", "two",
    "three", "one", "first", "large", "different", "effect", "effects", "outcome", "outcomes",
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | _EXTRA_STOP


# --------------------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------------------
class Client:
    """Small polite HTTP client for NCBI E-utilities and OpenAlex."""

    def __init__(self, email: str = "", ncbi_api_key: str = ""):
        self.email = email
        self.key = ncbi_api_key
        self.s = requests.Session()
        self.s.headers["User-Agent"] = f"journal-finder/1.0 ({email or 'no-email'})"
        self._last = 0.0

    def _throttle(self):
        gap = 0.11 if self.key else 0.35  # NCBI: 10 req/s with key, 3 without
        wait = self._last + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def ncbi(self, endpoint: str, params: dict, post: bool = False) -> requests.Response:
        p = {"tool": "journal-finder", **params}
        if self.email:
            p["email"] = self.email
        if self.key:
            p["api_key"] = self.key
        url = f"{EUTILS}/{endpoint}.fcgi"
        for attempt in range(5):
            self._throttle()
            r = self.s.post(url, data=p, timeout=90) if post else self.s.get(url, params=p, timeout=90)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        r.raise_for_status()
        return r

    def openalex(self, path: str, params: dict) -> dict:
        p = dict(params)
        if self.email:
            p["mailto"] = self.email
        for attempt in range(5):
            r = self.s.get(f"{OPENALEX}/{path}", params=p, timeout=60)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return {}


# --------------------------------------------------------------------------------------
# Query building
# --------------------------------------------------------------------------------------
def extract_keywords(text: str, n: int = 12) -> list[str]:
    """Most frequent informative words (and frequent 2-word phrases) in the abstract."""
    words = [w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower()) if w not in STOPWORDS]
    bigrams = Counter(f"{a} {b}" for a, b in zip(words, words[1:]))
    top_bi = [b for b, c in bigrams.most_common(4) if c >= 2]
    uni = Counter(w for w in words if len(w) > 3)
    covered = {w for b in top_bi for w in b.split()}
    top_uni = [w for w, _ in uni.most_common(n * 2) if w not in covered][: max(0, n - len(top_bi))]
    return top_bi + top_uni


def keywords_to_query(keywords: list[str]) -> str:
    return " OR ".join(f'"{k}"[tiab]' for k in keywords)


def build_query(
    terms_query: str,
    domains: list[str],
    types: list[str],
    year_from: int,
    year_to: int,
    english: bool = True,
    humans: bool = False,
    include_methods_journals: bool = False,
) -> str:
    parts = [f"({terms_query})"]
    if domains:
        dom = " OR ".join(f"({DOMAINS[d]})" for d in domains)
        if include_methods_journals:
            dom += f" OR ({METHODS_MESH})"
        parts.append(f"({dom})")
    if types:
        parts.append("(" + " OR ".join(f"({ARTICLE_TYPES[t]})" for t in types) + ")")
    parts.append(f"{year_from}:{year_to}[dp]")
    parts.append("hasabstract")
    if english:
        parts.append("english[la]")
    if humans:
        parts.append("humans[mh]")
    return " AND ".join(parts)


# --------------------------------------------------------------------------------------
# PubMed retrieval
# --------------------------------------------------------------------------------------
def esearch(client: Client, query: str, retmax: int = 300) -> tuple[int, list[str]]:
    r = client.ncbi(
        "esearch",
        {"db": "pubmed", "term": query, "retmax": retmax, "sort": "relevance", "retmode": "json"},
    )
    res = r.json()["esearchresult"]
    return int(res.get("count", 0)), res.get("idlist", [])


def _text(el) -> str:
    return "" if el is None else "".join(el.itertext()).strip()


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for art in root.iter("PubmedArticle"):
        mc = art.find("MedlineCitation")
        if mc is None:
            continue
        journal = mc.find("Article/Journal")
        issns = [e.text.strip() for e in mc.findall("Article/Journal/ISSN") if e.text]
        link = mc.findtext("MedlineJournalInfo/ISSNLinking")
        year = mc.findtext("Article/Journal/JournalIssue/PubDate/Year") or (
            (mc.findtext("Article/Journal/JournalIssue/PubDate/MedlineDate") or "")[:4]
        )
        out.append(
            {
                "pmid": mc.findtext("PMID"),
                "title": _text(mc.find("Article/ArticleTitle")),
                "abstract": " ".join(_text(a) for a in mc.findall("Article/Abstract/AbstractText")),
                "journal": (journal.findtext("Title") if journal is not None else "") or "",
                "journal_abbrev": (journal.findtext("ISOAbbreviation") if journal is not None else "") or "",
                "issn_l": (link or "").strip(),
                "issns": issns,
                "year": year,
                "pub_types": [e.text for e in mc.findall("Article/PublicationTypeList/PublicationType")],
            }
        )
    return out


def efetch(client: Client, pmids: list[str], batch: int = 200) -> list[dict]:
    recs: list[dict] = []
    for i in range(0, len(pmids), batch):
        r = client.ncbi(
            "efetch", {"db": "pubmed", "id": ",".join(pmids[i : i + batch]), "retmode": "xml"}, post=True
        )
        recs.extend(parse_pubmed_xml(r.text))
    return recs


def refine_keywords(abstract: str, recs: list[dict], n: int = 12) -> list[str]:
    """Pick the abstract's most *distinctive* terms relative to a first batch of PubMed hits
    (high frequency in your abstract, rare in the field)."""
    docs = [f"{r['title']}. {r['abstract']}" for r in recs] + [abstract]
    # min_df=2: a phrase must occur in at least one PubMed hit too (drops "psychiatry frequently")
    vec = TfidfVectorizer(stop_words=list(STOPWORDS), ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b")
    X = vec.fit_transform(docs)
    row = X[-1].toarray().ravel()
    vocab = np.array(vec.get_feature_names_out())
    picked: list[str] = []
    for idx in row.argsort()[::-1]:
        term = vocab[idx]
        if row[idx] <= 0 or len(picked) >= n:
            break
        if any(term in p or p in term for p in picked):  # avoid "missing" + "missing data"
            continue
        picked.append(term)
    return picked


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def journal_key(rec: dict) -> str:
    return rec["issn_l"] or (rec["issns"][0] if rec["issns"] else rec["journal"].lower())


def score_articles(query_text: str, recs: list[dict]) -> pd.DataFrame:
    """Cosine similarity (TF-IDF, uni+bigrams) between the query and each candidate."""
    docs = [f"{r['title']}. {r['abstract']}" for r in recs]
    vec = TfidfVectorizer(stop_words=list(STOPWORDS), ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X = vec.fit_transform(docs + [query_text])
    sims = cosine_similarity(X[-1], X[:-1]).ravel()
    df = pd.DataFrame(recs)
    df["similarity"] = sims
    df["jkey"] = [journal_key(r) for r in recs]
    return df.sort_values("similarity", ascending=False).reset_index(drop=True)


def score_journals(articles: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    top = articles.head(top_k)
    g = top.groupby("jkey")
    j = pd.DataFrame(
        {
            "Journal": g["journal"].first(),
            "Abbrev": g["journal_abbrev"].first(),
            "issn_l": g["issn_l"].first(),
            "issns": g["issns"].apply(lambda s: sorted({i for lst in s for i in lst})),
            "Similar articles": g.size(),
            "fit_raw": g["similarity"].sum(),
            "Best similarity": g["similarity"].max(),
            "pmids": g["pmid"].apply(list),
        }
    ).reset_index()
    j["Fit"] = 100 * j["fit_raw"] / j["fit_raw"].max() if len(j) else []
    return j.sort_values("Fit", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# OpenAlex journal metadata
# --------------------------------------------------------------------------------------
def openalex_sources(client: Client, issns: list[str]) -> dict[str, dict]:
    """Return {issn: source-summary} for every ISSN OpenAlex knows."""
    issns = sorted({i for i in issns if re.fullmatch(r"\d{4}-\d{3}[\dXx]", i or "")})
    found: dict[str, dict] = {}
    this_year = time.gmtime().tm_year
    for i in range(0, len(issns), 50):
        data = client.openalex(
            "sources", {"filter": "issn:" + "|".join(issns[i : i + 50]), "per_page": 200}
        )
        for s in data.get("results", []):
            stats = s.get("summary_stats") or {}
            cby = {c["year"]: c.get("works_count", 0) for c in s.get("counts_by_year") or []}
            recent = [cby.get(y, 0) for y in range(this_year - 3, this_year)]
            sub_counts: Counter = Counter()
            for t in s.get("topics") or []:
                name = (t.get("subfield") or {}).get("display_name")
                if name:
                    sub_counts[name] += t.get("count", 1)
            info = {
                "OpenAlex name": s.get("display_name"),
                "openalex_id": s.get("id"),
                "2-yr mean citedness": stats.get("2yr_mean_citedness"),
                "h-index": stats.get("h_index"),
                "Articles/yr": float(np.mean(recent)) if any(recent) else None,
                "Total works": s.get("works_count"),
                "Open access journal": bool(s.get("is_oa")),
                "In DOAJ": bool(s.get("is_in_doaj")),
                "APC (USD)": s.get("apc_usd"),
                "Publisher": s.get("host_organization_name"),
                "Homepage": s.get("homepage_url"),
                "Main subfield": sub_counts.most_common(1)[0][0] if sub_counts else None,
            }
            for issn in (s.get("issn") or []) + ([s.get("issn_l")] if s.get("issn_l") else []):
                prev = found.get(issn.upper())
                # Several OpenAlex records can share an ISSN; keep the biggest (the real journal).
                if prev is None or (info["Total works"] or 0) > (prev["Total works"] or 0):
                    found[issn.upper()] = info
    return found


def attach_openalex(journals: pd.DataFrame, sources: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for _, r in journals.iterrows():
        cands = [sources[i.upper()] for i in [r["issn_l"], *r["issns"]] if i and i.upper() in sources]
        # print & electronic ISSNs can map to different OpenAlex records; keep the largest
        rows.append(max(cands, key=lambda c: c["Total works"] or 0) if cands else {})
    meta = pd.DataFrame(rows, index=journals.index)
    return pd.concat([journals, meta], axis=1)


# --------------------------------------------------------------------------------------
# One-call convenience
# --------------------------------------------------------------------------------------
def find_journals(
    client: Client,
    abstract: str,
    terms_query: str,
    domains: list[str],
    types: list[str],
    year_from: int,
    year_to: int,
    english: bool = True,
    humans: bool = False,
    include_methods_journals: bool = False,
    n_candidates: int = 300,
    top_k: int = 50,
    refine: bool = True,
) -> dict:
    """If refine=True, a first pass with `terms_query` is used to learn which of the abstract's
    terms are distinctive, and the search is re-run with those (two-pass, JANE-like)."""
    args = (domains, types, year_from, year_to, english, humans, include_methods_journals)
    query = build_query(terms_query, *args)
    count, pmids = esearch(client, query, n_candidates)
    if not pmids:
        return {"query": query, "count": count, "articles": pd.DataFrame(),
                "journals": pd.DataFrame(), "keywords": []}
    recs = [r for r in efetch(client, pmids) if r["abstract"]]
    keywords: list[str] = []
    if refine and recs:
        keywords = refine_keywords(abstract, recs)
        q2 = build_query(keywords_to_query(keywords), *args)
        count2, pmids2 = esearch(client, q2, n_candidates)
        if pmids2:
            seen = {r["pmid"] for r in recs}
            new = [p for p in pmids2 if p not in seen]
            recs += [r for r in efetch(client, new) if r["abstract"]]
            query, count = q2, count2
    articles = score_articles(abstract, recs)
    journals = score_journals(articles, top_k)
    issns = [i for lst in journals["issns"] for i in lst] + journals["issn_l"].tolist()
    journals = attach_openalex(journals, openalex_sources(client, issns))
    return {"query": query, "count": count, "articles": articles, "journals": journals,
            "keywords": keywords}
