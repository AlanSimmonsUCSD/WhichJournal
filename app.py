"""
Journal Finder — Streamlit UI.   Run:  streamlit run app.py
"""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import core

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
    "— none —": None,
    "Statistical RCT methods paper → psychiatry": dict(
        domains=["Psychiatry"], types=["Randomized controlled trial", "Statistical / methods paper"],
        methods=True),
    "Systematic review / meta-analysis": dict(types=["Systematic review", "Meta-analysis"], methods=False),
    "Narrative review": dict(types=["Narrative review"], methods=False),
    "Original research (novel data)": dict(types=["Novel data (original research)"], methods=False),
}

EXAMPLE = (
    "Randomized controlled trials in psychiatry frequently suffer from high placebo response, missing "
    "data due to dropout, and heterogeneous treatment effects. We propose a Bayesian hierarchical mixed "
    "model for repeated measures that jointly models dropout and depression rating scale trajectories, "
    "and compare it to MMRM and multiple imputation in simulation studies and a re-analysis of "
    "antidepressant trials."
)


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def apply_preset():
    p = PRESETS[st.session_state.preset]
    if not p:
        return
    if "domains" in p:
        st.session_state.domains = p["domains"]
    st.session_state.types = p["types"]
    st.session_state.methods = p["methods"]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def run_search(email, key, abstract, terms, domains, types, y0, y1, english, humans, methods, n, k, refine):
    client = core.Client(email=email, ncbi_api_key=key)
    return core.find_journals(client, abstract, terms, list(domains), list(types), y0, y1, english,
                              humans, methods, n, k, refine)


# ------------------------------------------------------------------ sidebar: toggles
st.session_state.setdefault("domains", ["Psychiatry"])
st.session_state.setdefault("types", ["Randomized controlled trial", "Statistical / methods paper"])
st.session_state.setdefault("methods", True)

with st.sidebar:
    st.header("Filters")
    st.selectbox("Preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.multiselect("Domain(s)", list(core.DOMAINS), key="domains",
                   help="Restricts the PubMed search with MeSH terms. Empty = all of PubMed.")
    st.multiselect("Article type(s) to compare against", list(core.ARTICLE_TYPES), key="types",
                   help="Journals are scored on papers of these types. Combined with OR.")
    st.toggle("Also include methods / biostatistics journals", key="methods",
              help="Adds papers indexed with statistics / research-design MeSH terms even if they "
                   "fall outside the chosen domain (e.g. Statistics in Medicine, Trials).")
    this_year = dt.date.today().year
    y0, y1 = st.slider("Publication years", 2000, this_year, (this_year - 6, this_year))
    english = st.toggle("English only", True)
    humans = st.toggle("Human studies only", False)

    st.subheader("Journal filters")
    oa_only = st.toggle("Open-access journals only", False)
    doaj_only = st.toggle("DOAJ-listed only (screens out most predatory journals)", False)
    max_apc = st.number_input("Max APC (USD, 0 = no limit)", 0, 15000, 0, step=500)
    min_sim = st.slider("Min. similar articles per journal", 1, 10, 1)

    st.subheader("Plot")
    impact_metric = st.radio("Impact (x-axis)", ["2-yr mean citedness", "h-index"],
                             help="2-yr mean citedness is OpenAlex's free analogue of the Journal "
                                  "Impact Factor.")
    log_x = st.toggle("Log x-axis", True)
    size_metric = st.radio("Bubble size", ["Articles/yr", "Similar articles"])

    with st.expander("Advanced"):
        n_cand = st.slider("PubMed candidates per pass", 100, 1000, 300, step=100)
        top_k = st.slider("Top-K most similar articles used for scoring", 20, 200, 50, step=10)
        refine = st.toggle("Two-pass keyword refinement", True)
        email = st.text_input("Your e-mail (polite API use)", secret("CONTACT_EMAIL"))
        ncbi_key = st.text_input("NCBI API key (optional)", secret("NCBI_API_KEY"), type="password")

# ------------------------------------------------------------------ main
st.title("📚 Journal Finder")
st.caption("Paste a title + abstract. Journals are ranked by how similar their recent papers are "
           "to yours (fit) and plotted against impact. Data: PubMed (NCBI) + OpenAlex.")

abstract = st.text_area("Title and abstract", EXAMPLE, height=180)
auto_kw = core.extract_keywords(abstract) if abstract.strip() else []
terms = st.text_input("Search keywords (first pass — edit if they look off)",
                      core.keywords_to_query(auto_kw))

if st.button("Find journals", type="primary", disabled=not abstract.strip()):
    with st.spinner("Searching PubMed and OpenAlex…"):
        try:
            st.session_state.result = run_search(
                email, ncbi_key, abstract, terms, tuple(st.session_state.domains),
                tuple(st.session_state.types), y0, y1, english, humans, st.session_state.methods,
                n_cand, top_k, refine)
        except Exception as e:  # network / API errors
            st.error(f"Search failed: {e}")

res = st.session_state.get("result")
if not res:
    st.stop()

j = res["journals"].copy()
if j.empty:
    st.warning(f"No PubMed hits for this combination of filters (query below). Loosen a filter.")
    st.code(res["query"])
    st.stop()

# group for colour
dom_sub = set().union(*(DOMAIN_SUBFIELDS[d] for d in st.session_state.domains)) if st.session_state.domains else set()
j["Group"] = np.select(
    [j["Main subfield"].isin(dom_sub), j["Main subfield"].isin(METHODS_SUBFIELDS)],
    ["Selected domain", "Methods / statistics"], "Other")

# journal filters
j = j[j["Similar articles"] >= min_sim]
if oa_only:
    j = j[j["Open access journal"] == True]  # noqa: E712
if doaj_only:
    j = j[j["In DOAJ"] == True]  # noqa: E712
if max_apc:
    j = j[j["APC (USD)"].isna() | (j["APC (USD)"] <= max_apc)]

plot_df = j.dropna(subset=[impact_metric]).copy()
plot_df = plot_df[plot_df[impact_metric] > 0] if log_x else plot_df
# log-scaled bubble area so megajournals (PLOS ONE: ~17k papers/yr) don't swamp everything
plot_df["_size"] = np.log10(plot_df[size_metric].fillna(plot_df[size_metric].median()).clip(lower=1) + 9) ** 2
top_labels = set(plot_df.nlargest(10, "Fit")["Journal"])
plot_df["label"] = [a or n if n in top_labels else "" for a, n in zip(plot_df["Abbrev"], plot_df["Journal"])]

c1, c2, c3 = st.columns(3)
c1.metric("PubMed matches for query", f"{res['count']:,}")
c2.metric("Articles scored", len(res["articles"]))
c3.metric("Journals shown", len(j))

fig = px.scatter(
    plot_df, x=impact_metric, y="Fit", size="_size", size_max=38, color="Group", symbol="Group",
    color_discrete_map=GROUP_COLORS, symbol_map=GROUP_SYMBOLS, text="label", log_x=log_x,
    hover_name="Journal",
    hover_data={"_size": False, "label": False, "Group": False, "Fit": ":.0f",
                impact_metric: ":.2f", "Similar articles": True, "Articles/yr": ":.0f",
                "Main subfield": True, "Open access journal": True, "APC (USD)": True,
                "Publisher": True},
    labels={"Fit": "Fit (similarity to your abstract, 0–100)"},
    category_orders={"Group": list(GROUP_COLORS)},
)
fig.update_traces(textposition="top center", textfont_size=11,
                  marker=dict(opacity=0.8, line=dict(width=2, color="white")))
fig.update_layout(height=620, legend_title_text="", margin=dict(t=30, r=10, b=10, l=10),
                  legend=dict(orientation="h", y=1.04, x=0))
fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", rangemode="tozero")
st.plotly_chart(fig, width="stretch")
st.caption("Top-right = good fit and high impact. Bubble size = " + size_metric.lower() +
           " (log-scaled). Colour/shape = the journal's main OpenAlex subfield.")

# table
cols = ["Journal", "Group", "Fit", "Similar articles", "2-yr mean citedness", "h-index", "Articles/yr",
        "Main subfield", "Open access journal", "In DOAJ", "APC (USD)", "Publisher", "Homepage"]
st.subheader("Ranked journals")
st.dataframe(j[cols], hide_index=True, width="stretch",
             column_config={"Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%.0f"),
                            "Articles/yr": st.column_config.NumberColumn(format="%.0f"),
                            "Homepage": st.column_config.LinkColumn()})
st.download_button("Download CSV", j.drop(columns=["issns", "pmids"]).to_csv(index=False),
                   "journal_fit.csv", "text/csv")

with st.expander("Most similar articles (why each journal scored)"):
    a = res["articles"].head(top_k)[["similarity", "journal", "year", "title", "pmid"]].copy()
    a["link"] = "https://pubmed.ncbi.nlm.nih.gov/" + a["pmid"] + "/"
    st.dataframe(a.drop(columns="pmid"), hide_index=True, width="stretch",
                 column_config={"link": st.column_config.LinkColumn(),
                                "similarity": st.column_config.NumberColumn(format="%.3f")})

with st.expander("PubMed query actually used"):
    if res.get("keywords"):
        st.write("Refined keywords:", ", ".join(res["keywords"]))
    st.code(res["query"], language="text")
