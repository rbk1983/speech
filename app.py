
# app.py  (Speech Intelligence Platform for Kristalina Georgieva)
# ---------------------------------------------------------------
# Features:
# - Smart search by keyword(s), Theme, Region (if available), Date range
# - Always sorts results by recency
# - Per-speech Key Points (3–5 salient sentences) and concise Snippet (1–2 sentences)
# - Quote Finder: Top 5 dated quotes containing your keyword, with links
# - Thematic Briefings: evolution summary, top phrases, top speeches
# - Messaging Timeline & Theme Distribution charts
# - Quick Compare: contrast two years on a topic (quotes, terms, volume)
# - Briefing Pack generator (download as HTML)
# - Private Notes per speech (download as JSON)
# - Lightweight “message drift” check between two periods
#
# Assumptions:
# - Dataset pickle present as `speeches_data_reclassified.pkl` (preferred) or `speeches_data.pkl`
#   with columns: title, date, link, location, transcript, and either `new_themes` or `themes`.
# - Optional column `geo_tags` for region filtering (e.g., ["Africa","Europe"]). If absent, the UI hides region filter.

import os, re, json, io, datetime
from typing import List, Tuple, Dict
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Utilities
# -----------------------------

BOILERPLATE_PATTERNS = [
    r"the imf press center is a password-protected site for working journalists\.?",
    r"sign up to receive free e-mail notices when new series and/or country items are posted on the imf website\.?",
    r"modify your profile\.?",
    r"about\s*imf’s work",
    r"resource[s]?",
    r"topics",
    r"imf at a glance",
]

@st.cache_data(show_spinner=False)
def load_df() -> pd.DataFrame:
    candidates = [
        os.path.join(os.getcwd(), "speeches_data_reclassified.pkl"),
        os.path.join(os.getcwd(), "speeches_data.pkl"),
    ]
    pkl_path = None
    for p in candidates:
        if os.path.exists(p):
            pkl_path = p
            break
    if not pkl_path:
        st.error("Could not find speeches data (.pkl). Please add `speeches_data_reclassified.pkl` or `speeches_data.pkl` to this repo.")
        st.stop()

    df = pd.read_pickle(pkl_path)

    # Normalise columns
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        df["date"] = pd.to_datetime("today")

    # Unified theme column
    if "new_themes" in df.columns and df["new_themes"].notna().any():
        df["themes_u"] = df["new_themes"].apply(lambda x: x if isinstance(x, list) else [])
    elif "themes" in df.columns:
        df["themes_u"] = df["themes"].apply(lambda x: x if isinstance(x, list) else [])
    else:
        df["themes_u"] = [[] for _ in range(len(df))]

    # Clean transcript boilerplate
    def clean_text(text: str) -> str:
        t = (text or "").replace("\xa0"," ").strip()
        t = re.sub(r"\s+", " ", t)
        low = t.lower()
        for pat in BOILERPLATE_PATTERNS:
            low = re.sub(pat, " ", low)
        low = re.sub(r"\s+", " ", low).strip()
        if low:
            low = low[0].upper() + low[1:]
        return low

    df["clean_transcript"] = df["transcript"].fillna("").map(clean_text)
    return df

def sentence_split(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    # Keep reasonable lengths
    sents = [s.strip() for s in sents if 40 <= len(s.strip()) <= 320]
    return sents[:800]

def key_sentences(text: str, n: int = 4) -> List[str]:
    sents = sentence_split(text)
    if not sents:
        return []
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sents)
    scores = (X.power(2).sum(axis=1)).A.ravel()
    idx = scores.argsort()[::-1]
    chosen, used = [], set()
    for i in idx:
        sent = sents[i]
        sig = re.sub(r"[^a-z0-9]", "", sent.lower())[:80]
        if sig in used:
            continue
        used.add(sig)
        chosen.append(sent)
        if len(chosen) >= n:
            break
    return chosen

def concise_summary(text: str, max_sents: int = 2) -> str:
    pts = key_sentences(text, n=max_sents)
    return " ".join(pts) if pts else text[:240]

@st.cache_resource(show_spinner=False)
def build_corpus(df: pd.DataFrame):
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(df["clean_transcript"].tolist())
    return vec, X

def top_quotes(query: str, df: pd.DataFrame, k: int = 5) -> List[Tuple[str,str,str]]:
    if not query.strip():
        return []
    q_terms = [w for w in re.split(r"\W+", query.lower()) if w]
    quotes = []
    for _, row in df.iterrows():
        for s in sentence_split(row["clean_transcript"]):
            low = s.lower()
            if any(w in low for w in q_terms):
                quotes.append((s.strip(), row["date"], row["link"]))
    if not quotes:
        return []
    def score(item):
        s, d, _ = item
        hits = sum(1 for w in q_terms if w in s.lower())
        return (hits, len(s))
    quotes.sort(key=score, reverse=True)
    out, used = [], set()
    for s, d, link in quotes:
        sig = re.sub(r"[^a-z0-9]", "", s.lower())[:100]
        if sig in used:
            continue
        used.add(sig)
        out.append((s, d.strftime("%b %Y"), link))
        if len(out) >= k:
            break
    return out

def top_terms(texts: List[str], n: int = 15) -> List[Tuple[str,float]]:
    if not texts:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vec.fit_transform(texts)
    scores = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    pairs = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:n]
    return pairs

def drift_score(texts_a: List[str], texts_b: List[str]) -> float:
    # Cosine similarity between aggregated topic vectors; lower = more drift
    if not texts_a or not texts_b:
        return 0.0
    vec = TfidfVectorizer(stop_words="english", max_features=3000)
    A = vec.fit_transform(texts_a).mean(axis=0)
    B = vec.transform(texts_b).mean(axis=0)
    sim = cosine_similarity(A, B).ravel()[0]
    return float(sim)

def build_briefing_html(topic: str, df: pd.DataFrame, quotes: List[Tuple[str,str,str]]) -> str:
    html = io.StringIO()
    html.write(f"<h2>Briefing: {topic or 'General'}</h2>")
    html.write("<p>Auto-generated from Kristalina Georgieva speeches (Aug 2022–Aug 2025).</p>")
    # Summary
    summaries = [concise_summary(t, 2) for t in df['clean_transcript'].tolist()]
    overview = " ".join(summaries[:5])
    html.write(f"<h3>Summary</h3><p>{overview}</p>")
    # Timeline
    df_y = df.copy()
    df_y['year'] = df_y['date'].dt.year
    counts = df_y.groupby('year')['title'].count().reset_index()
    html.write("<h3>Volume by Year</h3><ul>")
    for _, r in counts.iterrows():
        html.write(f"<li>{int(r['year'])}: {int(r['title'])} speeches</li>")
    html.write("</ul>")
    # Quotes
    html.write("<h3>Top Quotes</h3><ol>")
    for q, d, link in quotes:
        html.write(f"<li>“{q}” — <em>{d}</em>. <a href='{link}'>Speech link</a></li>")
    html.write("</ol>")
    # Top items
    html.write("<h3>Top Speeches</h3><ol>")
    for _, row in df.sort_values('date', ascending=False).head(5).iterrows():
        html.write(f"<li>{row['date'].strftime('%Y-%m-%d')} — {row['title']} (<a href='{row['link']}'>link</a>)</li>")
    html.write("</ol>")
    return html.getvalue()

# -----------------------------
# App
# -----------------------------

def main():
    st.set_page_config(page_title="Kristalina Speech Intelligence", layout="wide")
    st.title("Kristalina Georgieva — Speech Intelligence Platform")

    df = load_df()
    vec, X = build_corpus(df)

    # Sidebar filters
    st.sidebar.header("Filters")
    all_themes = sorted({t for lst in df["themes_u"] for t in lst})
    theme_sel = st.sidebar.multiselect("Theme", options=all_themes, default=[])

    # Region filter if available
    if "geo_tags" in df.columns and df["geo_tags"].notna().any():
        regions_all = sorted({t for lst in df["geo_tags"] for t in (lst if isinstance(lst, list) else [])})
        region_sel = st.sidebar.multiselect("Region (if present)", options=regions_all, default=[])
    else:
        region_sel = []

    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    query = st.text_input("Search by keyword/phrase (optional)")

    # Base result set
    results = df.copy()

    # Theme filter (works even if no query)
    if theme_sel:
        results = results[results["themes_u"].apply(lambda lst: any(t in theme_sel for t in lst))]

    # Region filter
    if region_sel:
        results = results[results["geo_tags"].apply(lambda lst: any(t in region_sel for t in (lst if isinstance(lst, list) else [])))]

    # Date filter
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        results = results[(results["date"].dt.date >= start) & (results["date"].dt.date <= end)]

    # Query relevance
    if query.strip():
        q_vec = vec.transform([query])
        sims = cosine_similarity(q_vec, X).ravel()
        score_series = pd.Series(sims, index=df.index, name="score")
        results = results.join(score_series, how="left")
        results = results[results["score"].fillna(0) > 0]

    # Always show newest first
    results = results.sort_values(by="date", ascending=False)

    # --- Dashboard cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Speeches", len(results))
    c2.metric("Themes", len(all_themes))
    years = sorted(results["date"].dt.year.unique().tolist())
    c3.metric("Years Covered", f"{years[0]}–{years[-1]}" if years else "—")
    c4.metric("With Query", "Yes" if query.strip() else "No")

    # --- Charts ---
    with st.expander("Analytics (Theme distribution & Timeline)", expanded=True):
        # Theme distribution
        theme_counts = pd.Series([t for sub in results["themes_u"] for t in (sub if isinstance(sub, list) else [])]).value_counts().reset_index()
        theme_counts.columns = ["Theme", "Count"]
        if not theme_counts.empty:
            chart1 = alt.Chart(theme_counts).mark_bar().encode(
                x=alt.X("Count:Q"),
                y=alt.Y("Theme:N", sort='-x'),
                tooltip=["Theme","Count"]
            ).properties(height=300)
            st.altair_chart(chart1, use_container_width=True)

        # Timeline
        tl = results.copy()
        tl["Month"] = tl["date"].dt.to_period("M").dt.to_timestamp()
        timeline = tl.groupby("Month")["title"].count().reset_index()
        timeline.columns = ["Month","Speeches"]
        if not timeline.empty:
            chart2 = alt.Chart(timeline).mark_area().encode(
                x="Month:T",
                y="Speeches:Q",
                tooltip=["Month","Speeches"]
            ).properties(height=200)
            st.altair_chart(chart2, use_container_width=True)

    # --- Top quotes ---
    if query.strip():
        quotes = top_quotes(query, results, k=5)
        if quotes:
            st.subheader("Top Quotes Containing Your Keyword")
            for s, nice_date, link in quotes:
                st.markdown(f"- “{s}” — *{nice_date}*. [Speech link]({link})")

    # --- Results list ---
    st.subheader("Results (newest first)")
    for i, (_, row) in enumerate(results.iterrows(), start=1):
        with st.expander(f"{row['date'].strftime('%Y-%m-%d')} — {row['title']}"):
            # Meta
            themes_txt = ", ".join(row["themes_u"]) if row["themes_u"] else "N/A"
            st.markdown(f"**Themes:** {themes_txt}")
            if isinstance(row.get("location", None), str) and row["location"].strip():
                st.markdown(f"**Location:** {row['location']}")
            st.markdown(f"**Link:** [Open Speech]({row['link']})")

            # Key points & Snippet
            bullets = key_sentences(row["clean_transcript"], n=4)
            if bullets:
                st.markdown("**Key Points:**")
                for b in bullets:
                    st.markdown(f"- {b}")
            st.markdown("**Snippet:** " + concise_summary(row["clean_transcript"], max_sents=2))

            # Notes (stored in session only)
            note_key = f"note_{i}"
            cur_note = st.text_area("Private note (saved in this session only)", value=st.session_state.get(note_key,""))
            st.session_state[note_key] = cur_note

            with st.expander("Full Transcript (cleaned)"):
                st.write(row["clean_transcript"])

    # --- Quick Compare ---
    st.header("Quick Compare")
    years_all = sorted(df['date'].dt.year.unique().tolist())
    colA, colB = st.columns(2)
    year_a = colA.selectbox("Year A", options=years_all, index=0 if years_all else None)
    year_b = colB.selectbox("Year B", options=years_all, index=min(1, len(years_all)-1) if len(years_all)>1 else 0)
    cmp_theme = st.selectbox("Focus Theme (optional)", options=["(All)"] + all_themes, index=0)

    def filter_year_theme(y):
        sub = df[df['date'].dt.year == y]
        if cmp_theme != "(All)":
            sub = sub[sub['themes_u'].apply(lambda lst: cmp_theme in lst)]
        return sub

    if years_all:
        A = filter_year_theme(year_a)
        B = filter_year_theme(year_b)
        st.write(f"**{year_a}** speeches: {len(A)} | **{year_b}** speeches: {len(B)}")
        # Term drift
        sim = drift_score(A['clean_transcript'].tolist(), B['clean_transcript'].tolist())
        st.write(f"**Message similarity (0–1):** {sim:.2f} (lower = more drift)")
        # Top terms
        col1, col2 = st.columns(2)
        col1.write(f"**Top terms {year_a}**")
        for t, _ in top_terms(A['clean_transcript'].tolist(), 12):
            col1.write(f"- {t}")
        col2.write(f"**Top terms {year_b}**")
        for t, _ in top_terms(B['clean_transcript'].tolist(), 12):
            col2.write(f"- {t}")

    # --- Briefing Pack ---
    st.header("Briefing Pack")
    brief_topic = st.text_input("Briefing Topic", value=query if query else "")
    subset = results.copy()
    brief_quotes = top_quotes(brief_topic, subset, k=5) if brief_topic.strip() else []
    if st.button("Generate Briefing Pack (HTML)"):
        html = build_briefing_html(brief_topic, subset, brief_quotes)
        st.download_button("Download Briefing.html", data=html, file_name="Briefing.html", mime="text/html")

    # --- Export ---
    st.header("Export")
    # Export filtered results
    export_cols = ['title', 'date', 'link', 'location', 'themes_u']
    export_cols = [c for c in export_cols if c in results.columns]
    csv = results[export_cols].to_csv(index=False)
    st.download_button("Download results as CSV", data=csv, file_name="speeches_filtered.csv", mime="text/csv")

    # Export notes
    notes = {k:v for k,v in st.session_state.items() if k.startswith("note_") and v}
    if notes:
        st.download_button("Download my notes (JSON)", data=json.dumps(notes, indent=2), file_name="my_notes.json", mime="application/json")


if __name__ == "__main__":
    main()
