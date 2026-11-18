# 📚 Journal Finder

A JANE-style ([jane.biosemantics.org](https://jane.biosemantics.org/)) journal recommender with a
**fit × impact** scatter plot. Paste your title and abstract. The app finds the most similar recent
PubMed papers in the domains and article types you choose, then ranks journals by how well their
papers match yours. Each journal is plotted with:

| Axis | Meaning | Source |
|---|---|---|
| **y — Balanced score (0–100)** *(default)* | Fit and appetite combined (see below) | PubMed |
| *or* **Fit (0–100)** | Sum of TF-IDF cosine similarities of the journal's papers among the top-K most similar articles, scaled so the best journal = 100 | PubMed |
| *or* **Appetite (%)** | Share of all the journal's PubMed papers (same years) that are your article type, or your domain if no type is chosen | PubMed |
| **x — Impact** | 2-year mean citedness (free analogue of the Journal Impact Factor) or h-index | OpenAlex |
| **Bubble size** | Articles per year (3-yr mean) or # similar articles | OpenAlex / PubMed |
| **Colour + shape** | Journal's main subfield: in your domain / methods & statistics / other | OpenAlex |

![screenshot](screenshot.png)

## Features

- **Paste, upload or demo.** Paste a title and abstract, or upload a manuscript (**PDF, DOCX or
  DOC**). The title and abstract are pulled out and put in the text box for you to check. Files are
  read in memory and never stored. **🎲 Demo** loads a random recent PubMed abstract.
- **Fit × impact scatter plot.** Click a bubble (or use *Journal details*) for a card with the
  journal homepage, its NLM Catalog entry, an author-guidelines search, its OpenAlex page, and the
  similar papers it published.
- **Matching**: keyword (TF-IDF), semantic (sentence embeddings) or hybrid. See below.
- **Appetite and balanced score** (optional, off by default).
- **🧑‍🔬 Suggested reviewers**: authors of the most similar papers, weighted by similarity. First and
  last authors count fully, middle authors half. Includes their latest affiliation and PubMed links,
  plus a box to exclude yourself and your co-authors. Always check for conflicts of interest.
- **🔗 Shareable links**: a URL that restores all the filters, and optionally the abstract
  (compressed), which re-runs the search when opened. Leave the abstract out for unpublished work.
- **NIH public access status** for each journal, plus a filter (see below).
- **Impact bands**: soft shading on the plot and a table column splitting the journals into thirds
  (low / mid / high) by the x-axis impact metric. The cut points are the tertiles of *all journals in
  the current result set*, worked out before journal filters run so a journal keeps its band when
  you filter. They're relative guidance, not a global ranking.
- Ranked table plus CSV downloads for journals and reviewers.

## NIH Public Access Policy (2024, zero embargo)

Since 1 July 2025, NIH-funded papers must be freely available in PubMed Central **on the date of
publication**, with no embargo. The app checks each journal against the
[PMC Journal List](https://pmc.ncbi.nlm.nih.gov/journals/), downloaded from
`https://cdn.ncbi.nlm.nih.gov/pmc/home/jlist.csv` and cached for a day:

| Status | Meaning |
|---|---|
| ✅ **Immediate – all articles** | Active agreement, 0-month release delay: every article goes to PMC on publication. Compliant with no action from you. |
| ✅ **Immediate – NIH-funded** | Same, for NIH-funded articles. Compliant with no action from you. |
| ⏳ **Delayed** | Deposits in PMC, but after an embargo. Not compliant by itself: deposit your accepted manuscript via NIHMS (if the publisher allows zero embargo) or pay for open access. |
| ℹ️ **Not in PMC list** | No deposit agreement. Deposit your accepted manuscript via NIHMS yourself, after checking that the publisher allows zero-embargo deposit (fully open-access journals always do). |

The **"NIH immediate release only"** filter keeps the two ✅ groups. Journals outside them can still
be fine for NIH-funded work, but you have to check the publisher's policy. Several large publishers
now require paid open access for zero-embargo compliance. Your library's public-access guide is the
best source for publisher-level policies.

If the list can't be downloaded, the status shows *Unknown* and the filter is skipped with a warning.
As a backup you can save the CSV as `pmc_jlist.csv` next to `app.py`, but remember it goes stale.

## Semantic matching

Keyword matching only sees shared words. "Dropout" and "attrition" don't match, for example.
Semantic matching uses a small sentence-embedding model (`BAAI/bge-small-en-v1.5`) that compares
meaning. **It works with the current GitHub + Streamlit Community Cloud setup:**

- It runs through `fastembed` (ONNX Runtime), so there is **no PyTorch**. Installing it adds about
  60 MB, compared with more than 700 MB for `sentence-transformers` + torch.
- The model (about 130 MB) downloads from Hugging Face the first time someone uses semantic matching
  (about 30 s) and is then cached until the app restarts. Memory use stays well within the free
  tier's limits.
- Embedding about 600 candidate abstracts on the free tier's CPU takes roughly 5–15 s.
- If the model can't be loaded (no internet, `fastembed` not installed), the app falls back to
  keyword matching and says so.
- **Hybrid** averages the two scores after scaling each to 0–1. It's usually the most robust choice.

Want a biomedical-specific model (for example a PubMedBERT variant)? Those need PyTorch, which is
heavier but may still fit. Change `core.EMBED_MODEL` if fastembed supports the model; otherwise you'd
switch to `sentence-transformers`.

## Run it locally

```bash
git clone https://github.com/<you>/journal-finder.git
cd journal-finder
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your e-mail + API keys
streamlit run app.py
```

## Toggles

**Search scope (what papers your abstract is compared against)**

- **Domain(s)**: Cardiology, Cell & Molecular Neuroscience, Endocrinology & Diabetes, Immunology,
  Infectious Diseases, Neurology, Oncology, Psychiatry. These are broad MeSH filters, defined in
  `core.DOMAINS`.
- **Article type(s)** (combined with OR, using PubMed publication types):
  - Novel data (original research): journal articles, excluding reviews, meta-analyses, editorials, letters and comments
  - Randomized controlled trial
  - Clinical trial (any phase)
  - Observational study
  - Statistical / methods paper (statistics, research design and RCT-methodology MeSH)
  - Narrative review
  - Systematic review
  - Meta-analysis
  - Trial protocol
- **Also include methods / biostatistics journals**: adds stats-methods papers from outside your
  domain, so journals like *Statistics in Medicine*, *Trials* and *BMC Med Res Methodol* can appear.
- Publication-year range, English only, human studies only.

**Journal filters (applied after scoring)**: open-access only, DOAJ-listed only (a decent screen
against predatory journals), maximum APC in USD, and a minimum number of similar articles.

**Plot**: impact metric, log x-axis, bubble size metric.

**Paper-type presets**: Original research, Methods / statistics paper, Narrative review, and
Systematic review / meta-analysis. Each one fills in the article types (and the methods-journals
switch). Changing any of those by hand switches the preset back to *Custom*.

**🎲 Demo abstract** loads a random recent PubMed abstract that matches your selected domains and
types (or a random domain if none are selected) and sets the filters to match. The demo paper is
left out of the scoring, and the journal it was really published in is marked on the plot with a
★ and its rank, which is a quick way to sanity-check the tool.

Tip for a methods paper aimed at a clinical field: run it with the methods-journals switch **on**
and **off**. With it on, you see where methodologists publish. With it off, you see which
journals in the field publish this kind of work.

## Fit vs appetite vs balanced score

**Fit is a count, so it favours big journals.** A megajournal publishing 17,000 papers a year will
almost always have a few papers like yours. **Appetite is a share**: of everything the journal
publishes, how much is your kind of paper? A small stats journal with one similar paper but 75%
methods content may suit a methods paper better than a big clinical journal with three similar
papers that are rare exceptions.

**Balanced score** = weighted geometric mean of the two:

```
balanced = fit_scaled^(1 - w) × appetite_scaled^w        (rescaled so the best journal = 100)
```

- `fit_scaled` = fit / 100.
- `appetite_scaled` = appetite % on a log scale (shares range from about 0.1% to 100%), mapped to 0.05–1.
  The 0.05 floor means a low share penalises a journal but never zeroes it.
- `w` is the *Balance* slider: 0 = pure fit, 0.5 = equal weight (default), 1 = pure appetite.
- A geometric mean means a journal must do reasonably on **both** to score well.

Appetite costs two PubMed count queries per journal, so it is computed for the top 25 journals by
fit. That takes about 10–25 s without an NCBI key and about 5 s with one. Neither measure knows
editorial policy: many journals only publish reviews by invitation, so check the author guidelines.

## How it works

1. A first PubMed query is built from your abstract's keywords + domain + type filters. Results are
   sorted by PubMed "Best Match".
2. **Two-pass refinement**: using those hits as a background corpus, the app picks the terms that are
   most *distinctive* in your abstract and searches again.
3. All candidate abstracts are scored by TF-IDF cosine similarity to yours. The top K (default 50)
   decide each journal's fit.
4. Journals are matched to OpenAlex by ISSN to get impact, volume, OA status and APC.

Code layout:
- `core.py`: search, scoring, appetite and reviewers. It has no UI code, so you can use it from a notebook.
- `textextract.py`: reads PDF / DOCX / DOC files.
- `app.py`: the Streamlit UI.
- `packages.txt`: system packages for Streamlit Cloud (`antiword`, for old .doc files).
  To read .doc files locally, run `brew install antiword` or `apt install antiword`, or just save
  the file as .docx.

## Put it on GitHub / online

```bash
git init && git add . && git commit -m "Journal Finder"
gh repo create journal-finder --public --source=. --push     # or create the repo on github.com and push
```

To get a free public web app, go to [share.streamlit.io](https://share.streamlit.io), choose
**New app**, pick the repo and set `app.py` as the main file. Paste `CONTACT_EMAIL` /
`OPENALEX_API_KEY` / `NCBI_API_KEY` under *Settings → Secrets*. **Never commit `.streamlit/secrets.toml`**; it is
already in `.gitignore`.

## Data & licensing notes

- **PubMed / E-utilities**: free. Please supply an e-mail address. Limits are 3 requests/s without
  an API key and 10/s with one ([NCBI usage policy](https://www.ncbi.nlm.nih.gov/books/NBK25497/)).
- **OpenAlex**: CC0, free to use and redistribute. **Get a free API key**
  ([help.openalex.org/api/authentication](https://help.openalex.org/api/authentication/)).
  Without one, requests draw on a small daily allowance shared by everyone on the same IP address.
  On Streamlit Cloud that means sharing it with other people's apps, so impact data can disappear.
  The app still shows fit rankings when that happens, and it caches journal lookups for a week to
  keep usage low.
- The official **Journal Impact Factor** (Clarivate) is proprietary, so don't scrape it or commit it
  to a public repo. That is why this app uses OpenAlex's 2-yr mean citedness. It follows a similar
  idea but the numbers are not identical.

## Limitations

- Fit is lexical (TF-IDF), not semantic. Edit the first-pass keywords if they look off.
  Sentence-embedding models (for example `sentence-transformers`) would be the natural upgrade.
- A journal's fit depends on how many similar papers it published, so high-volume journals get a
  boost (JANE has the same effect). Use the "Similar articles" column to judge this.
- "Main subfield" is OpenAlex's topic classification and is occasionally odd for broad journals.
