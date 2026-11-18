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

import io
import os
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
# [majr] = major topic of the paper. Plain [mh] is far too broad here: ordinary trials get tagged
# "Research Design" sub-terms like "Double-Blind Method", which made 20-40% of clinical journals
# look like methods journals.
METHODS_MESH = (
    '"Models, Statistical"[majr] OR "Data Interpretation, Statistical"[majr] OR '
    '"Statistics as Topic"[majr] OR "Research Design"[majr] OR '
    '"Randomized Controlled Trials as Topic"[majr] OR "Bayes Theorem"[majr] OR '
    '"Computer Simulation"[majr] OR "Sample Size"[majr]'
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
class OpenAlexLimit(RuntimeError):
    """OpenAlex refused because the (shared or personal) daily budget is spent."""


class Client:
    """Small polite HTTP client for NCBI E-utilities and OpenAlex."""

    def __init__(self, email: str = "", ncbi_api_key: str = "", openalex_api_key: str = ""):
        self.email = email
        self.key = ncbi_api_key
        self.openalex_key = openalex_api_key
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
        headers = {"Authorization": f"Bearer {self.openalex_key}"} if self.openalex_key else {}
        for attempt in range(5):
            r = self.s.get(f"{OPENALEX}/{path}", params=p, headers=headers, timeout=60)
            if r.status_code == 429 and float(r.headers.get("retry-after", 0) or 0) > 30:
                raise OpenAlexLimit(
                    "OpenAlex's daily allowance is used up"
                    + (" for your API key." if self.openalex_key else
                       " (no API key, so the allowance is shared by everyone on this server's IP)."))
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


def _authors(mc) -> list[dict]:
    out = []
    for a in mc.findall("Article/AuthorList/Author"):
        last = a.findtext("LastName")
        if not last:  # collective/group author
            continue
        out.append({"last": last.strip(), "fore": (a.findtext("ForeName") or "").strip(),
                    "initials": (a.findtext("Initials") or "").strip(),
                    "affiliation": (a.findtext("AffiliationInfo/Affiliation") or "").strip()})
    return out


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
                "authors": _authors(mc),
            }
        )
    return out


def efetch(client: Client, pmids: list[str], batch: int = 200) -> list[dict]:
    recs: list[dict] = []
    for i in range(0, len(pmids), batch):
        for attempt in range(3):  # NCBI occasionally returns truncated XML; just retry
            r = client.ncbi(
                "efetch", {"db": "pubmed", "id": ",".join(pmids[i : i + batch]), "retmode": "xml"},
                post=True,
            )
            try:
                recs.extend(parse_pubmed_xml(r.text))
                break
            except ET.ParseError:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
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


EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # ~130 MB ONNX model, runs on CPU, no PyTorch needed
_EMBEDDER = None


def get_embedder():
    """Lazily load the sentence-embedding model (downloaded from Hugging Face on first use)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from fastembed import TextEmbedding  # optional dependency
        _EMBEDDER = TextEmbedding(EMBED_MODEL)
    return _EMBEDDER


def tfidf_similarity(query_text: str, docs: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(stop_words=list(STOPWORDS), ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X = vec.fit_transform(docs + [query_text])
    return cosine_similarity(X[-1], X[:-1]).ravel()


def embedding_similarity(query_text: str, docs: list[str]) -> np.ndarray:
    V = np.array(list(get_embedder().embed([query_text] + docs, batch_size=32)), dtype=np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return V[1:] @ V[0]


def _minmax(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.ones_like(x)


def score_articles(query_text: str, recs: list[dict], method: str = "tfidf") -> pd.DataFrame:
    """Similarity between the query and each candidate.
    method: "tfidf" (keyword overlap), "semantic" (sentence embeddings) or "hybrid" (average of
    both after min-max scaling). Falls back to TF-IDF if the embedding model can't be loaded."""
    docs = [f"{r['title']}. {r['abstract']}" for r in recs]
    used = "tfidf"
    sims = tfidf_similarity(query_text, docs)
    if method in ("semantic", "hybrid"):
        try:
            emb = embedding_similarity(query_text, docs)
            if method == "semantic":
                sims = np.clip(emb, 0, None)
            else:
                sims = 0.5 * _minmax(sims) + 0.5 * _minmax(emb)
            used = method
        except Exception:  # model missing / download blocked -> keyword matching
            pass
    df = pd.DataFrame(recs)
    df["similarity"] = sims
    df.attrs["method_used"] = used
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
# Reviewer suggestions (JANE-style "find authors")
# --------------------------------------------------------------------------------------
def suggest_reviewers(articles: pd.DataFrame, top_k: int = 50, n: int = 30) -> pd.DataFrame:
    """Authors of the most similar articles, scored by the summed similarity of their papers.
    First/last authors get full weight, middle authors half (they're often less senior/central)."""
    if articles.empty or "authors" not in articles:
        return pd.DataFrame()
    people: dict[str, dict] = {}
    for _, a in articles.head(top_k).iterrows():
        auths = a["authors"] or []
        for pos, au in enumerate(auths):
            key = f"{au['last']} {au['initials'] or au['fore'][:1]}".lower()
            w = 1.0 if pos in (0, len(auths) - 1) else 0.5
            p = people.setdefault(key, {"Name": f"{au['fore']} {au['last']}".strip(), "last": au["last"],
                                        "initials": au["initials"], "score": 0.0, "papers": [],
                                        "first/last": 0, "affiliation": "", "aff_year": ""})
            p["score"] += w * a["similarity"]
            p["papers"].append(a["pmid"])
            p["first/last"] += pos in (0, len(auths) - 1)
            if au["affiliation"] and str(a["year"]) >= str(p["aff_year"]):
                p["affiliation"], p["aff_year"] = au["affiliation"], a["year"]
    if not people:
        return pd.DataFrame()
    df = pd.DataFrame(people.values())
    df["Similar papers"] = df["papers"].str.len()
    df["Score"] = 100 * df["score"] / df["score"].max()
    df["PubMed"] = ("https://pubmed.ncbi.nlm.nih.gov/?term=" + df["last"].str.replace(" ", "+")
                    + "+" + df["initials"] + "%5Bau%5D")
    df["Their similar papers"] = df["papers"].apply(
        lambda ps: "https://pubmed.ncbi.nlm.nih.gov/?term=" + "+OR+".join(f"{p}%5Bpmid%5D" for p in ps))
    df = df.rename(columns={"affiliation": "Latest affiliation"})
    return (df.sort_values(["Score", "Similar papers"], ascending=False).head(n)
              [["Name", "Score", "Similar papers", "first/last", "Latest affiliation", "PubMed",
                "Their similar papers"]].reset_index(drop=True))


# --------------------------------------------------------------------------------------
# OpenAlex journal metadata
# --------------------------------------------------------------------------------------
_SOURCE_CACHE: dict[str, tuple[float, dict | None]] = {}  # issn -> (fetched_at, info or None)
_CACHE_TTL = 7 * 24 * 3600


def openalex_sources(client: Client, issns: list[str]) -> dict[str, dict]:
    """Return {issn: source-summary} for every ISSN OpenAlex knows. Results are cached for a week
    in this process, so repeat searches cost (almost) no OpenAlex budget."""
    issns = sorted({i.upper() for i in issns if re.fullmatch(r"\d{4}-\d{3}[\dXx]", i or "")})
    now = time.time()
    fresh = {i: _SOURCE_CACHE[i][1] for i in issns
             if i in _SOURCE_CACHE and now - _SOURCE_CACHE[i][0] < _CACHE_TTL}
    found: dict[str, dict] = {i: v for i, v in fresh.items() if v}
    issns = [i for i in issns if i not in fresh]
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
        for issn in issns[i : i + 50]:
            _SOURCE_CACHE[issn] = (now, found.get(issn))
    return found


META_COLS = ["OpenAlex name", "openalex_id", "2-yr mean citedness", "h-index", "Articles/yr",
             "Total works", "Open access journal", "In DOAJ", "APC (USD)", "Publisher", "Homepage",
             "Main subfield"]


def attach_openalex(journals: pd.DataFrame, sources: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for _, r in journals.iterrows():
        cands = [sources[i.upper()] for i in [r["issn_l"], *r["issns"]] if i and i.upper() in sources]
        # print & electronic ISSNs can map to different OpenAlex records; keep the largest
        rows.append(max(cands, key=lambda c: c["Total works"] or 0) if cands else {})
    meta = pd.DataFrame(rows, index=journals.index).reindex(columns=META_COLS)
    return pd.concat([journals, meta], axis=1)


# --------------------------------------------------------------------------------------
# Appetite: what share of a journal's output is "your kind" of paper?
# --------------------------------------------------------------------------------------
_COUNT_CACHE: dict[str, int] = {}


def pubmed_count(client: Client, query: str) -> int:
    if query not in _COUNT_CACHE:
        r = client.ncbi("esearch", {"db": "pubmed", "term": query, "retmax": 0, "retmode": "json"})
        _COUNT_CACHE[query] = int(r.json()["esearchresult"].get("count", 0))
    return _COUNT_CACHE[query]


def appetite_clause(types: list[str], domains: list[str]) -> str:
    """What counts as 'your kind of paper': the chosen article types, else the chosen domains."""
    if types:
        return "(" + " OR ".join(f"({ARTICLE_TYPES[t]})" for t in types) + ")"
    if domains:
        return "(" + " OR ".join(f"({DOMAINS[d]})" for d in domains) + ")"
    return ""


def add_appetite(client: Client, journals: pd.DataFrame, types: list[str], domains: list[str],
                 year_from: int, year_to: int, max_journals: int = 25) -> pd.DataFrame:
    """Appetite % = journal's PubMed papers matching your types (or domains) / all its PubMed
    papers, over the same years. Computed for the top `max_journals` by fit (2 queries each)."""
    j = journals.copy()
    j["Your-type papers"] = np.nan
    j["Total papers"] = np.nan
    j["Appetite %"] = np.nan
    clause = appetite_clause(types, domains)
    if not clause:
        return j
    years = f"{year_from}:{year_to}[dp]"
    for idx in j.index[:max_journals]:
        issn = j.at[idx, "issn_l"] or (j.at[idx, "issns"][0] if j.at[idx, "issns"] else "")
        if not issn:
            continue
        base = f"{issn}[is] AND {years}"
        total = pubmed_count(client, base)
        if total:
            mine = pubmed_count(client, f"{base} AND {clause}")
            j.loc[idx, ["Your-type papers", "Total papers", "Appetite %"]] = [mine, total, 100 * mine / total]
    return j


def balanced_score(j: pd.DataFrame, weight: float = 0.5) -> pd.Series:
    """Weighted geometric mean of fit and appetite, both scaled 0-1, result 0-100.
    weight = 0 -> pure fit, 0.5 -> equal balance, 1 -> pure appetite.
    Appetite is put on a log scale first (shares span 0.1%-100%) and floored at 0.05 so a
    journal is penalised, not zeroed, for a low share."""
    fit = (j["Fit"] / 100).clip(lower=0.01)
    app = j["Appetite %"]
    if app.notna().sum() == 0:
        return pd.Series(np.nan, index=j.index)
    la = np.log10(app.clip(lower=0.01))
    lo, hi = la.min(), la.max()
    app_n = ((la - lo) / (hi - lo) if hi > lo else la * 0 + 1).clip(lower=0) * 0.95 + 0.05
    score = 100 * fit ** (1 - weight) * app_n ** weight
    return score / score.max() * 100


# --------------------------------------------------------------------------------------
# NIH Public Access Policy (2024, zero embargo): PMC Journal List
# --------------------------------------------------------------------------------------
PMC_JLIST_URL = "https://cdn.ncbi.nlm.nih.gov/pmc/home/jlist.csv"
_PMC_CACHE: dict = {}

NIH_STATUS_HELP = {
    "Immediate – all articles": "Journal deposits the final version of every article in PMC on "
                                "publication. NIH-compliant with no action from you.",
    "Immediate – NIH-funded": "Journal deposits NIH-funded articles in PMC on publication. "
                              "NIH-compliant with no action from you.",
    "Delayed": "Journal deposits in PMC, but only after an embargo. Not compliant by itself: you "
               "must deposit your accepted manuscript via NIHMS (if the publisher allows zero "
               "embargo) or pay for open access.",
    "Not in PMC list": "No PMC deposit agreement. You must deposit your accepted manuscript via "
                       "NIHMS yourself; check the publisher allows zero-embargo deposit (fully "
                       "open-access journals always do).",
}


def _col(df: pd.DataFrame, *names: str) -> str | None:
    """Find a column by any of several names (the CSV headers changed in 2024)."""
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        for lc, c in low.items():
            if lc.startswith(n.lower()):
                return c
    return None


def _delay_months(v) -> float:
    v = str(v or "").lower()
    if "immediate" in v:
        return 0.0
    m = re.search(r"(\d+)", v)
    return float(m.group(1)) if m else np.nan


def parse_pmc_jlist(csv_text: str) -> dict[str, dict]:
    """{ISSN: {'NIH / PMC': status, 'PMC delay': text, 'PMC URL': url}} from the PMC Journal List."""
    df = pd.read_csv(io.StringIO(csv_text), dtype=str).fillna("")
    c_p = _col(df, "ISSN (print)", "pISSN")
    c_e = _col(df, "ISSN (online)", "eISSN")
    c_delay = _col(df, "Release Delay", "Free access")
    c_status = _col(df, "Agreement Status", "Deposit status")
    c_agree = _col(df, "Agreement to Deposit", "Participation level")
    c_url = _col(df, "PMC URL", "Journal URL")
    if not (c_delay and (c_p or c_e)):
        raise ValueError("PMC Journal List format not recognised")
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        status = (r[c_status] if c_status else "").strip()
        active = status == "" or status.lower() == "active"
        if not active:
            continue  # ceased / predecessor titles
        months = _delay_months(r[c_delay])
        agree = (r[c_agree] if c_agree else "").lower()
        if months == 0:
            label = "Immediate – NIH-funded" if "nih" in agree else "Immediate – all articles"
        else:
            label = "Delayed"
        info = {"NIH / PMC": label, "PMC delay": r[c_delay].strip(),
                "PMC URL": r[c_url].strip() if c_url else ""}
        for c in (c_p, c_e):
            issn = (r[c] if c else "").strip().upper()
            if issn:
                prev = out.get(issn)
                if prev is None or prev["NIH / PMC"] == "Delayed":  # prefer an immediate record
                    out[issn] = info
    return out


def pmc_journal_list(client: Client, ttl: float = 24 * 3600) -> dict[str, dict]:
    """Download (and cache for a day) the PMC Journal List."""
    if _PMC_CACHE.get("data") is not None and time.time() - _PMC_CACHE["t"] < ttl:
        return _PMC_CACHE["data"]
    local = os.environ.get("PMC_JLIST_FILE", "pmc_jlist.csv")  # optional manual copy / backup
    if os.path.exists(local):
        with open(local, encoding="utf-8-sig", errors="replace") as f:
            text = f.read()
    else:
        r = client.s.get(PMC_JLIST_URL, timeout=60)
        r.raise_for_status()
        text = r.content.decode("utf-8-sig", errors="replace")
    data = parse_pmc_jlist(text)
    _PMC_CACHE.update(data=data, t=time.time())
    return data


def attach_pmc(journals: pd.DataFrame, pmc: dict[str, dict] | None) -> pd.DataFrame:
    j = journals.copy()
    rank = {"Immediate – all articles": 0, "Immediate – NIH-funded": 1, "Delayed": 2}
    labels, delays, urls = [], [], []
    for _, r in j.iterrows():
        if pmc is None:
            labels.append("Unknown"); delays.append(""); urls.append("")
            continue
        cands = [pmc[i.upper()] for i in [r["issn_l"], *r["issns"]] if i and i.upper() in pmc]
        best = min(cands, key=lambda c: rank[c["NIH / PMC"]]) if cands else None
        labels.append(best["NIH / PMC"] if best else "Not in PMC list")
        delays.append(best["PMC delay"] if best else "")
        urls.append(best["PMC URL"] if best else "")
    j["NIH / PMC"], j["PMC delay"], j["PMC URL"] = labels, delays, urls
    return j


# --------------------------------------------------------------------------------------
# One-call convenience
# --------------------------------------------------------------------------------------
def random_article(
    client: Client,
    domains: list[str],
    types: list[str],
    year_from: int,
    year_to: int,
    rng=None,
) -> dict | None:
    """A random PubMed article (with abstract) matching the domain/type/year filters.
    PubMed only lets us page through the first 10,000 hits, so we sample within those."""
    import random

    rng = rng or random.Random()
    if not domains:  # no domain chosen -> pick one at random so demos vary
        domains = [rng.choice(list(DOMAINS))]
    parts = ["(" + " OR ".join(f"({DOMAINS[d]})" for d in domains) + ")"]
    if types:
        parts.append("(" + " OR ".join(f"({ARTICLE_TYPES[t]})" for t in types) + ")")
    parts += [f"{year_from}:{year_to}[dp]", "hasabstract", "english[la]"]
    q = " AND ".join(parts)
    r = client.ncbi("esearch", {"db": "pubmed", "term": q, "retmax": 0, "retmode": "json"})
    count = int(r.json()["esearchresult"].get("count", 0))
    if not count:
        return None
    for _ in range(3):
        start = rng.randrange(min(count, 9999))
        r = client.ncbi("esearch", {"db": "pubmed", "term": q, "retmax": 1, "retstart": start,
                                    "retmode": "json"})
        ids = r.json()["esearchresult"].get("idlist", [])
        recs = efetch(client, ids) if ids else []
        if recs and len(recs[0]["abstract"]) > 400:
            return {**recs[0], "demo_domains": domains}
    return None


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
    exclude_pmids: tuple[str, ...] = (),
    appetite: bool = False,
    method: str = "tfidf",
) -> dict:
    """If refine=True, a first pass with `terms_query` is used to learn which of the abstract's
    terms are distinctive, and the search is re-run with those (two-pass, JANE-like)."""
    args = (domains, types, year_from, year_to, english, humans, include_methods_journals)
    query = build_query(terms_query, *args)
    count, pmids = esearch(client, query, n_candidates)
    if not pmids:
        return {"query": query, "count": count, "articles": pd.DataFrame(),
                "journals": pd.DataFrame(), "keywords": [], "warning": "",
                "method_used": method, "reviewers": pd.DataFrame()}
    excl = set(exclude_pmids)
    recs = [r for r in efetch(client, [p for p in pmids if p not in excl]) if r["abstract"]]
    keywords: list[str] = []
    if refine and recs:
        keywords = refine_keywords(abstract, recs)
        q2 = build_query(keywords_to_query(keywords), *args)
        count2, pmids2 = esearch(client, q2, n_candidates)
        if pmids2:
            seen = {r["pmid"] for r in recs} | excl
            new = [p for p in pmids2 if p not in seen]
            recs += [r for r in efetch(client, new) if r["abstract"]]
            query, count = q2, count2
    articles = score_articles(abstract, recs, method)
    journals = score_journals(articles, top_k)
    issns = [i for lst in journals["issns"] for i in lst] + journals["issn_l"].tolist()
    warning = ""
    try:
        sources = openalex_sources(client, issns)
    except (OpenAlexLimit, requests.RequestException) as e:
        sources, warning = {}, f"Impact / size / open-access data unavailable: {e}"
    journals = attach_openalex(journals, sources)
    try:
        journals = attach_pmc(journals, pmc_journal_list(client))
    except Exception as e:  # list unreachable or format changed
        journals = attach_pmc(journals, None)
        warning = (warning + " " if warning else "") + f"PMC Journal List unavailable ({e}); NIH status unknown."
    if appetite:
        journals = add_appetite(client, journals, types, domains, year_from, year_to)
    return {"query": query, "count": count, "articles": articles, "journals": journals,
            "keywords": keywords, "warning": warning,
            "method_used": articles.attrs.get("method_used", "tfidf"),
            "reviewers": suggest_reviewers(articles, top_k)}
