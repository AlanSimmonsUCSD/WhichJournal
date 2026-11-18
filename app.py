"""
Journal Finder — Streamlit UI.   Run:  streamlit run app.py
"""
import base64
import datetime as dt
import random
import zlib
from urllib.parse import urlencode, urlsplit

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import core
import textextract

st.set_page_config(page_title="Journal Finder", page_icon="📚", layout="wide")

# OpenAlex subfield names that count as "in your domain" (used for colouring only)
DOMAIN_SUBFIELDS = {
    "Cardiology": {"Cardiology and Cardiovascular Medicine"},
    "Cell & Molecular Neuroscience": {"Cellular and Molecular Neuroscience", "Cognitive Neuroscience"},
    "Endocrinology & Diabetes": {"Endocrinology, Diabetes and Metabolism", "Endocrinology"},
    "Immunology": {"Immunology", "Immunology and Allergy"},
    "Infectious Diseases": {"Infectious Diseases"},
    "Neurology": {"Neurology", "Neurology (clinical)"},
    "Oncology": {"Oncology", "Cancer Research"},
    "Psychiatry": {"Psychiatry and Mental health", "Clinical Psychology"},
}
METHODS_SUBFIELDS = {"Statistics and Probability", "Statistics, Probability and Uncertainty",
                     "Epidemiology", "Health Informatics"}
GROUP_COLORS = {"Selected domain": "#2a78d6", "Methods / statistics": "#eb6834", "Other": "#1baf7a"}
GROUP_SYMBOLS = {"Selected domain": "circle", "Methods / statistics": "diamond", "Other": "square"}

PRESETS = {
    "Custom": None,
    "Original research (novel data)": dict(types=["Novel data (original research)"], methods=False),
    "Methods / statistics paper": dict(types=["Statistical / methods paper"], methods=True),
    "Narrative review": dict(types=["Narrative review"], methods=False),
    "Systematic review / meta-analysis": dict(types=["Systematic review", "Meta-analysis"], methods=False),
}
MATCHING = {"Keyword (TF-IDF, fast)": "tfidf", "Semantic (embeddings)": "semantic",
            "Hybrid (both)": "hybrid"}
Y_OPTIONS = ["Fit", "Balanced score", "Appetite %"]
Y_COLS = {"Fit": "Fit", "Balanced score": "Balanced", "Appetite %": "Appetite %"}
Y_LABELS = {"Fit": "Fit (similarity to your abstract, 0–100)",
            "Balanced": "Balanced score (fit × appetite, 0–100)",
            "Appetite %": "Appetite (% of journal's papers that are your type)"}
THIS_YEAR = dt.date.today().year

try:
    import fastembed  # noqa: F401
    HAS_EMBED = True
except ImportError:
    HAS_EMBED = False


def preset_for(pub_types: list[str]) -> str:
    """Which preset best describes a PubMed record (sets filters for the demo)."""
    pt = {p.lower() for p in pub_types}
    if pt & {"meta-analysis", "network meta-analysis", "systematic review"}:
        return "Systematic review / meta-analysis"
    if "review" in pt:
        return "Narrative review"
    return "Original research (novel data)"


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


# ------------------------------------------------------------------ state defaults
ss = st.session_state
DEFAULTS = dict(
    abstract="", preset="Custom", types=[], domains=[], methods=False,
    years=(THIS_YEAR - 5, THIS_YEAR), english=True, humans=False,
    oa_only=False, doaj_only=False, max_apc=0, min_sim=1,
    appetite=False, weight=0.5, matching="Keyword (TF-IDF, fast)",
    y_metric="Fit", impact_metric="2-yr mean citedness", log_x=True, size_metric="Articles/yr",
    demo=None, exclude_pmid="",
)
for k, v in DEFAULTS.items():
    ss.setdefault(k, v)


# ------------------------------------------------------------------ shareable links
def _pack(text: str) -> str:
    return base64.urlsafe_b64encode(zlib.compress(text.encode(), 9)).decode().rstrip("=")


def _unpack(s: str) -> str:
    return zlib.decompress(base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))).decode()


def state_to_params(include_abstract: bool) -> dict:
    p = {
        "types": "|".join(ss.types), "domains": "|".join(ss.domains),
        "years": f"{ss.years[0]}-{ss.years[1]}", "methods": int(ss.methods),
        "en": int(ss.english), "hum": int(ss.humans), "oa": int(ss.oa_only),
        "doaj": int(ss.doaj_only), "apc": ss.max_apc, "min": ss.min_sim,
        "app": int(ss.appetite), "w": ss.weight, "match": MATCHING[ss.matching],
        "y": ss.y_metric, "x": ss.impact_metric, "logx": int(ss.log_x), "size": ss.size_metric,
    }
    if include_abstract and ss.abstract.strip():
        p["abs"] = _pack(ss.abstract)
        if ss.demo:
            p["excl"] = ss.demo["pmid"]
    return {k: v for k, v in p.items() if v not in ("", None)}


def apply_query_params():
    """On first load, restore a shared search from the URL (and run it if it has an abstract)."""
    q = st.query_params
    if not q:
        return
    try:
        lst = lambda k, allowed: [x for x in q.get(k, "").split("|") if x in allowed]  # noqa: E731
        ss.types = lst("types", core.ARTICLE_TYPES)
        ss.domains = lst("domains", core.DOMAINS)
        if "years" in q:
            a, b = (int(x) for x in q["years"].split("-"))
            ss.years = (max(2000, a), min(THIS_YEAR, b))
        for key, name in [("methods", "methods"), ("en", "english"), ("hum", "humans"),
                          ("oa", "oa_only"), ("doaj", "doaj_only"), ("app", "appetite"),
                          ("logx", "log_x")]:
            if key in q:
                ss[name] = q[key] == "1"
        if "apc" in q:
            ss.max_apc = int(q["apc"])
        if "min" in q:
            ss.min_sim = int(q["min"])
        if "w" in q:
            ss.weight = float(q["w"])
        inv = {v: k for k, v in MATCHING.items()}
        if q.get("match") in inv:
            ss.matching = inv[q["match"]]
        if q.get("y") in Y_OPTIONS:
            ss.y_metric = q["y"]
        if q.get("x") in ("2-yr mean citedness", "h-index"):
            ss.impact_metric = q["x"]
        if q.get("size") in ("Articles/yr", "Similar articles"):
            ss.size_metric = q["size"]
        if "abs" in q:
            ss.abstract = _unpack(q["abs"])
            ss.exclude_pmid = q.get("excl", "")
            ss.autorun = True
    except Exception:
        ss.link_error = "That link couldn't be fully read — some settings were left at defaults."


if "qp_done" not in ss:
    ss.qp_done = True
    apply_query_params()


# ------------------------------------------------------------------ callbacks
def apply_preset():
    p = PRESETS[ss.preset]
    if p:
        ss.types, ss.methods = p["types"], p["methods"]


def mark_custom():
    ss.preset = "Custom"


def clear_all():
    ss.abstract, ss.demo, ss.result, ss.exclude_pmid = "", None, None, ""
    st.query_params.clear()


def load_demo():
    """Pull a random recent PubMed abstract (matching the chosen domains/types, if any)."""
    client = core.Client(secret("CONTACT_EMAIL"), secret("NCBI_API_KEY"))
    try:
        rec = core.random_article(client, list(ss.domains), list(ss.types), THIS_YEAR - 3, THIS_YEAR,
                                  random.Random())
    except Exception as e:
        ss.notice = ("warning", f"Couldn't fetch a demo abstract: {e}")
        return
    if not rec:
        ss.notice = ("warning", "No article matched the current filters — try clearing them.")
        return
    ss.notice = None
    ss.abstract = f"{rec['title']}\n\n{rec['abstract']}"
    ss.demo, ss.exclude_pmid, ss.result = rec, rec["pmid"], None
    if not ss.domains:
        ss.domains = rec["demo_domains"]
    if not ss.types:  # only auto-set the article type if the user hasn't chosen one
        ss.preset = preset_for(rec["pub_types"])
        apply_preset()


def load_upload():
    f = ss.get("upload")
    if f is None:
        return
    try:
        text = textextract.extract_text(f.getvalue(), f.name)
        title, abstract = textextract.find_title_abstract(text)
    except Exception as e:
        ss.notice = ("error", f"Couldn't read {f.name}: {e}")
        return
    if len(abstract) < 100:
        ss.notice = ("warning", f"Read {f.name}, but couldn't find an abstract — please paste it.")
        return
    ss.abstract = f"{title}\n\n{abstract}" if title else abstract
    ss.demo, ss.exclude_pmid, ss.result = None, "", None
    ss.notice = ("info", f"Loaded title and abstract from **{f.name}** — check the text below "
                         "(extraction is heuristic) and edit if needed. The file is not stored.")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def run_search(keys, abstract, terms, domains, types, y0, y1, english, humans, methods, n, k,
               refine, exclude, appetite, method):
    client = core.Client(*keys)
    return core.find_journals(client, abstract, terms, list(domains), list(types), y0, y1, english,
                              humans, methods, n, k, refine, exclude, appetite, method)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("What are you submitting?")
    st.selectbox("Paper type preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.multiselect("Article type(s) to compare against", list(core.ARTICLE_TYPES), key="types",
                   on_change=mark_custom, placeholder="Any type",
                   help="Journals are scored on papers of these types (combined with OR). "
                        "Empty = all types.")
    st.multiselect("Domain(s)", list(core.DOMAINS), key="domains", placeholder="All of PubMed",
                   help="Restricts the PubMed search with broad MeSH terms. Empty = no restriction.")
    st.toggle("Also include methods / biostatistics journals", key="methods", on_change=mark_custom,
              help="Adds papers whose main topic is statistics / research design, even outside the "
                   "chosen domain (e.g. Statistics in Medicine). Only applies when a domain is selected.")
    st.slider("Compare with papers published", 2000, THIS_YEAR, key="years")
    st.toggle("English only", key="english")
    st.toggle("Human studies only", key="humans")

    st.header("Journal filters")
    st.toggle("Open-access journals only", key="oa_only")
    st.toggle("DOAJ-listed only", key="doaj_only",
              help="Directory of Open Access Journals — screens out most predatory journals.")
    st.number_input("Max APC in USD (0 = no limit)", 0, 15000, step=500, key="max_apc")
    st.slider("Min. similar articles per journal", 1, 10, key="min_sim")

    st.header("Score")
    st.radio("Matching", list(MATCHING), key="matching",
             help="Keyword: shared (rare) words. Semantic: a small language model compares meaning, "
                  "so different wording for the same idea still matches (first use downloads a "
                  "~130 MB model, ~30 s). Hybrid averages the two.")
    if not HAS_EMBED and MATCHING[ss.matching] != "tfidf":
        st.caption("⚠️ `fastembed` isn't installed — keyword matching will be used.")
    st.toggle("Compute appetite (adds ~10–25 s)", key="appetite",
              help="For the top 25 journals: what share of ALL their PubMed papers (same years) are "
                   "your article type(s) — or your domain(s) if no type is chosen. Corrects fit's "
                   "bias toward very large journals.")
    st.slider("Balance:  fit ← → appetite", 0.0, 1.0, step=0.1, key="weight",
              help="0 = pure fit, 0.5 = equal weight (geometric mean), 1 = pure appetite.",
              disabled=not ss.appetite)

    st.header("Plot")
    st.radio("Y-axis", Y_OPTIONS, key="y_metric",
             help="Balanced score and Appetite % need 'Compute appetite'; otherwise Fit is shown.")
    st.radio("Impact (x-axis)", ["2-yr mean citedness", "h-index"], key="impact_metric",
             help="2-yr mean citedness is OpenAlex's free analogue of the Journal Impact Factor "
                  "(similar idea, different numbers).")
    st.toggle("Log x-axis", key="log_x")
    st.radio("Bubble size", ["Articles/yr", "Similar articles"], key="size_metric")

    with st.expander("Advanced"):
        terms_override = st.text_input(
            "Override first-pass keywords", "",
            help="Leave empty to derive keywords from the abstract automatically. Otherwise use "
                 "PubMed syntax, e.g. dropout[tiab] OR \"missing data\"[tiab]")
        n_cand = st.slider("PubMed candidates per pass", 100, 1000, 300, step=100)
        top_k = st.slider("Top-K most similar articles used for scoring", 20, 200, 50, step=10)
        refine = st.toggle("Two-pass keyword refinement", True)
        email = st.text_input("Your e-mail (polite API use)", secret("CONTACT_EMAIL"))
        ncbi_key = st.text_input("NCBI API key", secret("NCBI_API_KEY"), type="password")
        oa_key = st.text_input("OpenAlex API key", secret("OPENALEX_API_KEY"), type="password",
                               help="Free at openalex.org. Without one, impact data uses a small "
                                    "daily allowance shared by everyone on the same IP address.")

# ------------------------------------------------------------------ main: input
st.title("📚 Journal Finder")
st.caption("Paste or upload your manuscript's title and abstract. Journals are ranked by how "
           "similar their recent papers are to yours (**fit**) and plotted against **impact**. "
           "Data: PubMed (NCBI) + OpenAlex.")

if ss.get("link_error"):
    st.warning(ss.pop("link_error"))

st.file_uploader("Upload a manuscript (PDF, DOCX or DOC) to pull out its title and abstract",
                 type=["pdf", "docx", "doc"], key="upload", on_change=load_upload)
st.text_area("Title and abstract", key="abstract", height=200,
             placeholder="Paste your title and abstract here, upload a file above, or click 🎲 Demo "
                         "to load a random recent PubMed abstract.")
b1, b2, b3, _ = st.columns([1.2, 1.2, 0.8, 4])
go = b1.button("Find journals", type="primary", disabled=len(ss.abstract.strip()) < 100,
               width="stretch")
b2.button("🎲 Demo abstract", on_click=load_demo, width="stretch",
          help="Loads a random recent PubMed abstract (from your selected domains/types if any). "
               "The paper itself is excluded from scoring, and its real journal is marked on the plot.")
b3.button("Clear", on_click=clear_all, width="stretch")

if ss.get("notice"):
    kind, msg = ss.notice
    getattr(st, kind)(msg)
if ss.demo and not ss.abstract.startswith(ss.demo["title"]):
    ss.demo, ss.exclude_pmid = None, ""  # user replaced the demo text with their own
if ss.demo:
    d = ss.demo
    st.info(f"Demo paper: PMID [{d['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{d['pmid']}/) — "
            f"actually published in **{d['journal']}** ({d['year']}). "
            f"Types: {', '.join(d['pub_types'])}. Filters were set to match; press **Find journals**.")
if 0 < len(ss.abstract.strip()) < 100:
    st.caption("Paste at least ~100 characters — a full abstract works best.")

if go or ss.pop("autorun", False):
    ss.notice = None
    terms = terms_override.strip() or core.keywords_to_query(core.extract_keywords(ss.abstract))
    if not terms:
        st.error("Couldn't extract any keywords — please paste more text.")
        st.stop()
    exclude = (ss.exclude_pmid,) if ss.exclude_pmid else ()
    method = MATCHING[ss.matching] if HAS_EMBED else "tfidf"
    msg = "Searching PubMed and OpenAlex… (≈10–20 s"
    msg += ", +10–25 s for appetite" if ss.appetite else ""
    msg += ", first semantic run downloads a model" if method != "tfidf" else ""
    with st.spinner(msg + ")"):
        try:
            ss.result = res_ = run_search(
                (email, ncbi_key, oa_key), ss.abstract, terms, tuple(ss.domains), tuple(ss.types),
                ss.years[0], ss.years[1], ss.english, ss.humans, ss.methods, n_cand, top_k, refine,
                exclude, ss.appetite, method)
            if res_.get("warning"):
                run_search.clear()  # don't keep metadata-less results cached for a day
        except Exception as e:
            st.error(f"Search failed: {e}")
            ss.result = None

res = ss.get("result")
if not res:
    with st.expander("How it works", expanded=not ss.abstract):
        st.markdown(
            "1. Your abstract's keywords + the chosen domain/type filters build a PubMed query.\n"
            "2. A second pass searches again with the terms most *distinctive* to your abstract.\n"
            "3. Every candidate abstract gets a similarity score (keyword, semantic or hybrid); the "
            "top-K decide each journal's **fit** (best journal = 100).\n"
            "4. Optional **appetite**: the share of *all* a journal's PubMed papers that are your "
            "article type (or domain) — corrects fit's bias toward huge journals. **Balanced score** "
            "= weighted geometric mean of fit and appetite.\n"
            "5. Journals are matched to OpenAlex for impact, volume, open-access status and APC.\n"
            "6. Authors of the most similar papers are listed as possible **reviewers**.")
    st.stop()

# ------------------------------------------------------------------ results
j = res["journals"].copy()
if j.empty:
    st.warning("No PubMed hits for this combination of filters. Loosen a filter (domain, type, years).")
    with st.expander("PubMed query"):
        st.code(res["query"], language="text")
    st.stop()
if res.get("warning"):
    st.warning(res["warning"] + " Showing fit only. Fix: add a free OpenAlex API key "
               "(sidebar → Advanced, or the app's secrets), or try again after midnight UTC.")
wanted = MATCHING[ss.matching]
if wanted != "tfidf" and res.get("method_used") == "tfidf":
    st.info("Semantic matching wasn't available (model couldn't be loaded), so keyword matching "
            "was used.")

has_app = "Appetite %" in j and j["Appetite %"].notna().any()
j["Balanced"] = core.balanced_score(j, ss.weight) if has_app else np.nan
rank_col = "Balanced" if has_app else "Fit"
j = j.sort_values(rank_col, ascending=False, na_position="last").reset_index(drop=True)

dom_sub = set().union(*(DOMAIN_SUBFIELDS[d] for d in ss.domains)) if ss.domains else set()
j["Group"] = np.select(
    [j["Main subfield"].isin(dom_sub), j["Main subfield"].isin(METHODS_SUBFIELDS)],
    ["Selected domain", "Methods / statistics"], "Other")
j["NLM Catalog"] = [f"https://www.ncbi.nlm.nih.gov/nlmcatalog/?term={i or (s[0] if s else '')}%5BISSN%5D"
                    for i, s in zip(j["issn_l"], j["issns"])]

actual_key = None
if ss.demo:
    d_issns = {ss.demo["issn_l"], *ss.demo["issns"]} - {""}
    hit = j[j.apply(lambda r: bool(d_issns & ({r["issn_l"], *r["issns"]})), axis=1)]
    actual_key = hit["jkey"].iloc[0] if len(hit) else None

j = j[j["Similar articles"] >= ss.min_sim]
if ss.oa_only:
    j = j[j["Open access journal"] == True]  # noqa: E712
if ss.doaj_only:
    j = j[j["In DOAJ"] == True]  # noqa: E712
if ss.max_apc:
    j = j[j["APC (USD)"].isna() | (j["APC (USD)"] <= ss.max_apc)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("PubMed matches for query", f"{res['count']:,}")
c2.metric("Articles scored", len(res["articles"]))
c3.metric("Journals shown", len(j))
c4.metric("Matching", {"tfidf": "Keyword", "semantic": "Semantic", "hybrid": "Hybrid"}[res.get("method_used", "tfidf")])

y_col = Y_COLS[ss.y_metric] if has_app else "Fit"
if not has_app and ss.y_metric != "Fit":
    st.caption("Turn on **Compute appetite** to plot balanced score or appetite — showing fit.")
x_col = ss.impact_metric
plot_df = j.dropna(subset=[x_col, y_col]).copy()
if ss.log_x:
    plot_df = plot_df[plot_df[x_col] > 0]

selected_key = None
if plot_df.empty:
    top = j.head(20).iloc[::-1]
    fig = px.bar(top, x=rank_col, y="Journal", orientation="h", hover_data=["Similar articles", "Fit"],
                 color_discrete_sequence=[GROUP_COLORS["Selected domain"]])
    fig.update_layout(height=40 + 26 * len(top), margin=dict(t=10, r=10, b=10, l=10), yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{rank_col} (no impact data available). Top 20 journals.")
else:
    size_src = plot_df[ss.size_metric].fillna(plot_df[ss.size_metric].median()).fillna(1).clip(lower=1)
    plot_df["_size"] = np.log10(size_src + 9) ** 2  # log-scaled so megajournals don't swamp the plot
    top_labels = set(plot_df.nlargest(10, y_col)["jkey"]) | ({actual_key} if actual_key else set())
    plot_df["label"] = [(a or n) if k in top_labels else ""
                        for a, n, k in zip(plot_df["Abbrev"], plot_df["Journal"], plot_df["jkey"])]
    hover = {"_size": False, "label": False, "Group": False, "Fit": ":.0f", x_col: ":.2f",
             "Similar articles": True, "Articles/yr": ":.0f", "Main subfield": True,
             "Open access journal": True, "APC (USD)": True, "Publisher": True}
    if has_app:
        hover.update({"Balanced": ":.0f", "Appetite %": ":.1f"})
    fig = px.scatter(
        plot_df, x=x_col, y=y_col, size="_size", size_max=38, color="Group", symbol="Group",
        color_discrete_map=GROUP_COLORS, symbol_map=GROUP_SYMBOLS, text="label", log_x=ss.log_x,
        hover_name="Journal", hover_data=hover, custom_data=["jkey"], labels=Y_LABELS,
        category_orders={"Group": list(GROUP_COLORS)},
    )
    fig.update_traces(textposition="top center", textfont_size=11,
                      marker=dict(opacity=0.8, line=dict(width=2, color="white")))
    if actual_key and actual_key in set(plot_df["jkey"]):
        r = plot_df[plot_df["jkey"] == actual_key].iloc[0]
        fig.add_annotation(x=np.log10(r[x_col]) if ss.log_x else r[x_col], y=r[y_col],
                           text="★ actually published here", showarrow=True, arrowhead=2, ax=60,
                           ay=-40, font=dict(size=12))
    fig.update_layout(height=620, legend_title_text="", margin=dict(t=30, r=10, b=10, l=10),
                      legend=dict(orientation="h", y=1.04, x=0), clickmode="event+select")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", rangemode="tozero")
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points",
                            key="scatter")
    pts = (event or {}).get("selection", {}).get("points", []) if event else []
    if pts and pts[0].get("customdata"):
        selected_key = pts[0]["customdata"][0]
    st.caption(f"Top-right = high {Y_LABELS[y_col].split(' (')[0].lower()} and high impact. Bubble "
               f"size = {ss.size_metric.lower()} (log-scaled). **Click a bubble** for journal details "
               "and links.")

# ------------------------------------------------------------------ journal detail card
journal_names = j["Journal"].tolist()
pick = st.selectbox("Journal details", ["—"] + journal_names,
                    index=(journal_names.index(j.loc[j["jkey"] == selected_key, "Journal"].iloc[0]) + 1)
                    if selected_key in set(j["jkey"]) else 0)
if pick != "—":
    r = j[j["Journal"] == pick].iloc[0]
    with st.container(border=True):
        st.markdown(f"#### {r['Journal']}")
        m1, m2, m3, m4 = st.columns(4) if has_app else (*st.columns(3), None)
        m1.metric("Fit", f"{r['Fit']:.0f}")
        m2.metric("2-yr mean citedness", "—" if pd.isna(r["2-yr mean citedness"]) else f"{r['2-yr mean citedness']:.2f}")
        m3.metric("Articles/yr", "—" if pd.isna(r["Articles/yr"]) else f"{r['Articles/yr']:.0f}")
        if m4 is not None:
            m4.metric("Appetite", "—" if pd.isna(r.get("Appetite %", np.nan)) else f"{r['Appetite %']:.1f}%")
        facts = [f"Publisher: {r['Publisher']}" if isinstance(r["Publisher"], str) else "",
                 f"Main subfield: {r['Main subfield']}" if isinstance(r["Main subfield"], str) else "",
                 "Open-access journal" if r["Open access journal"] == True else "",  # noqa: E712
                 "In DOAJ" if r["In DOAJ"] == True else "",  # noqa: E712
                 f"APC: ${r['APC (USD)']:,.0f}" if pd.notna(r["APC (USD)"]) else ""]
        st.caption(" · ".join(f for f in facts if f))
        links = []
        if isinstance(r["Homepage"], str) and r["Homepage"]:
            links.append(f"[Journal homepage]({r['Homepage']})")
        links.append(f"[NLM Catalog]({r['NLM Catalog']})")
        q = (r["Journal"] + " author guidelines").replace(" ", "+")
        links.append(f"[Find author guidelines](https://duckduckgo.com/?q={q})")
        if isinstance(r["openalex_id"], str):
            links.append(f"[OpenAlex]({r['openalex_id']})")
        links.append("[Its similar papers](https://pubmed.ncbi.nlm.nih.gov/?term="
                     + "+OR+".join(f"{p}%5Bpmid%5D" for p in r["pmids"]) + ")")
        st.markdown(" · ".join(links))

if ss.demo:
    if actual_key and actual_key in set(j["jkey"]):
        rank = int(np.where(j["jkey"].values == actual_key)[0][0]) + 1
        st.success(f"The demo paper's real journal (**{ss.demo['journal']}**) ranked **#{rank}** of "
                   f"{len(j)} by {rank_col.lower()}.")
    elif actual_key:
        st.info("The demo paper's real journal was found but is hidden by your journal filters.")
    else:
        st.info(f"The demo paper's real journal (**{ss.demo['journal']}**) wasn't among the "
                "best-matching journals — a reminder that fit is only one input.")

# ------------------------------------------------------------------ table
st.subheader("Ranked journals")
cols = ["Journal", "Group", "Balanced", "Fit", "Appetite %", "Your-type papers", "Total papers",
        "Similar articles", "2-yr mean citedness", "h-index", "Articles/yr", "Main subfield",
        "Open access journal", "In DOAJ", "APC (USD)", "Publisher", "Homepage", "NLM Catalog"]
app_cols = {"Balanced", "Appetite %", "Your-type papers", "Total papers"}
cols = [c for c in cols if c in j and (c not in app_cols or has_app)]
st.dataframe(j[cols], hide_index=True, width="stretch",
             column_config={
                 "Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%.0f"),
                 "Balanced": st.column_config.ProgressColumn("Balanced", min_value=0, max_value=100,
                                                             format="%.0f"),
                 "Appetite %": st.column_config.NumberColumn(format="%.1f%%"),
                 "Your-type papers": st.column_config.NumberColumn(format="%d"),
                 "Total papers": st.column_config.NumberColumn(format="%d"),
                 "Articles/yr": st.column_config.NumberColumn(format="%.0f"),
                 "Homepage": st.column_config.LinkColumn(display_text="homepage"),
                 "NLM Catalog": st.column_config.LinkColumn(display_text="catalog")})
st.download_button("Download CSV", j.drop(columns=["issns", "pmids"]).to_csv(index=False),
                   "journal_fit.csv", "text/csv")

# ------------------------------------------------------------------ pull-downs
with st.expander("🧑‍🔬 Suggested reviewers"):
    rv = res.get("reviewers", pd.DataFrame())
    if rv is None or rv.empty:
        st.write("No author data available.")
    else:
        excl = st.text_input("Exclude names (you, co-authors, collaborators) — comma-separated "
                             "last names", "")
        if excl.strip():
            bad = {x.strip().lower() for x in excl.split(",") if x.strip()}
            rv = rv[~rv["Name"].str.lower().str.split().str[-1].isin(bad)]
        st.caption("Authors of the papers most similar to yours, weighted by similarity "
                   "(first/last authors count fully, middle authors half). Check for conflicts of "
                   "interest — recent co-authors, same institution — before suggesting anyone.")
        st.dataframe(rv, hide_index=True, width="stretch",
                     column_config={
                         "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                         "first/last": st.column_config.NumberColumn("First/last-author papers"),
                         "PubMed": st.column_config.LinkColumn("All their papers", display_text="PubMed"),
                         "Their similar papers": st.column_config.LinkColumn(display_text="view")})
        st.download_button("Download reviewers CSV", rv.to_csv(index=False), "reviewers.csv", "text/csv")

with st.expander("📄 Most similar articles (why each journal scored)"):
    a = res["articles"].head(top_k)[["similarity", "journal", "year", "title", "pmid"]].copy()
    a["link"] = "https://pubmed.ncbi.nlm.nih.gov/" + a["pmid"] + "/"
    st.dataframe(a.drop(columns="pmid"), hide_index=True, width="stretch",
                 column_config={"link": st.column_config.LinkColumn(display_text="PubMed"),
                                "similarity": st.column_config.NumberColumn(format="%.3f")})

with st.expander("🔗 Share this search"):
    inc = st.checkbox("Include the abstract in the link",
                      help="Off: the link restores your filters only. On: it also carries the "
                           "abstract (compressed) and re-runs the search when opened.")
    if inc:
        st.caption("⚠️ Anyone with the link can read the abstract, and it may end up in browser "
                   "history and server logs — don't do this for unpublished work you want kept private.")
    try:
        base = urlsplit(st.context.url)._replace(query="", fragment="").geturl()
    except Exception:
        base = ""
    link = f"{base}?{urlencode(state_to_params(inc))}"
    st.code(link, language="text")
    st.caption("Use the copy button on the right of the box.")

with st.expander("🔎 PubMed query actually used"):
    if res.get("keywords"):
        st.write("Refined keywords:", ", ".join(res["keywords"]))
    st.code(res["query"], language="text")
