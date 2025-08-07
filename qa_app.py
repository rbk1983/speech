
import os
import re
import datetime
from typing import List, Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Data loading
# -----------------------------

@st.cache_data(show_spinner=False)
def load_df() -> pd.DataFrame:
    # Prefer reclassified data if present
    candidates = [
        os.path.join(os.getcwd(), "speeches_data_reclassified.pkl"),
        os.path.join(os.getcwd(), "speeches_data.pkl"),
        os.path.join("/home/oai/share", "speeches_data_reclassified.pkl"),
        os.path.join("/home/oai/share", "speeches_data.pkl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_pickle(p)
            break
    else:
        raise FileNotFoundError("Could not find speeches data (.pkl).")

    # Normalise columns
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        df["date"] = pd.to_datetime("today")

    # Unified theme column
    if "new_themes" in df.columns:
        df["themes_u"] = df["new_themes"]
    elif "themes" in df.columns:
        df["themes_u"] = df["themes"]
    else:
        df["themes_u"] = [[] for _ in range(len(df))]

    # Clean transcript boilerplate up-front
    df["clean_transcript"] = df["transcript"].fillna("").map(clean_text)
    return df

# -----------------------------
# Cleaning & summarisation
# -----------------------------

BOILERPLATE_PATTERNS = [
    r"the imf press center is a password-protected site for working journalists\.?",
    r"sign up to receive free e-mail notices when new series and/or country items are posted on the imf website\.?",
    r"modify your profile\.?",
    r"kristalina georgieva",
    r"about\s*imf’s work",
    r"resource[s]?",
    r"topics",
    r"imf at a glance",
]

def clean_text(text: str) -> str:
    t = text.replace("\xa0"," ").strip()
    # remove repeated whitespace
    t = re.sub(r"\s+", " ", t)
    # remove boilerplate phrases
    low = t.lower()
    for pat in BOILERPLATE_PATTERNS:
        low = re.sub(pat, " ", low)
    # collapse whitespace again
    low = re.sub(r"\s+", " ", low).strip()
    # Capitalise first letter for display snippets
    if low:
        low = low[0].upper() + low[1:]
    return low

def sentence_split(text: str) -> List[str]:
    # simple sentence split
    sents = re.split(r"(?<=[.!?])\s+", text)
    # keep reasonable length
    sents = [s.strip() for s in sents if 30 <= len(s.strip()) <= 300]
    return sents[:500]

def key_sentences(text: str, n: int = 4) -> List[str]:
    """Return n salient sentences as bullet-point 'key points'."""
    sents = sentence_split(text)
    if not sents:
        return []
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sents)
    # Score each sentence by its TF-IDF norm (proxy for informativeness)
    scores = (X.power(2).sum(axis=1)).A.ravel()
    idx = scores.argsort()[::-1]
    chosen = []
    used = set()
    for i in idx:
        sent = sents[i]
        # de-duplicate by substrings
        sig = re.sub(r"[^a-z0-9]", "", sent.lower())[:60]
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

def build_corpus(df: pd.DataFrame) -> Tuple[TfidfVectorizer, any]:
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(df["clean_transcript"].tolist())
    return vec, X

def top_quotes(query: str, df: pd.DataFrame, k: int = 5) -> List[Tuple[str,str,str]]:
    """Return up to k best quotes (sentence-level) for query.
    Each quote: (quote_text, display_date 'Mon YYYY', link)
    """
    if not query.strip():
        return []
    q_terms = [w for w in re.split(r"\\W+", query.lower()) if w]
    quotes = []
    for _, row in df.iterrows():
        for s in sentence_split(row["clean_transcript"]):
            s_low = s.lower()
            if any(w in s_low for w in q_terms):
                quotes.append((s.strip(), row["date"], row["link"]))
    if not quotes:
        return []
    # Rank quotes by length and term presence
    def score(item):
        s, d, _ = item
        hits = sum(1 for w in q_terms if w in s.lower())
        return (hits, len(s))
    quotes.sort(key=score, reverse=True)
    # Format date
    out = []
    used = set()
    for s, d, link in quotes:
        sig = re.sub(r"[^a-z0-9]", "", s.lower())[:80]
        if sig in used:
            continue
        used.add(sig)
        out.append((s, d.strftime("%b %Y"), link))
        if len(out) >= k:
            break
    return out

# -----------------------------
# App
# -----------------------------

def main():
    st.set_page_config(page_title="IMF MD Speeches Explorer", layout="wide")
    st.title("Kristalina Georgieva Speeches (Aug 2022 – Aug 2025)")

    df = load_df()
    vec, X = build_corpus(df)

    # Sidebar filters
    st.sidebar.header("Filters")
    all_themes = sorted({t for lst in df["themes_u"] for t in lst})
    selected_themes = st.sidebar.multiselect("By Theme", options=all_themes, default=[])

    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    date_range = st.sidebar.date_input("By Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    query = st.text_input("Enter a topic or keyword (optional)")

    # Build results set:
    results = df.copy()

    # Apply theme filter even if query is empty (per user request)
    if selected_themes:
        results = results[results["themes_u"].apply(lambda lst: any(t in selected_themes for t in lst))]

    # Apply date filter
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        results = results[(results["date"].dt.date >= start) & (results["date"].dt.date <= end)]

    # Apply query relevance scoring if a query is provided
    if query.strip():
        q_vec = TfidfVectorizer(stop_words="english").fit(df["clean_transcript"]).transform([query])
        # Use cosine similarity against full-text matrix
        # (use same vocabulary by rebuilding vec to ensure compatibility)
        vec_q = TfidfVectorizer(stop_words="english")
        vec_q.fit(df["clean_transcript"])
        X_q = vec_q.transform(df["clean_transcript"])
        sims = cosine_similarity(q_vec, X_q).ravel()
        results = results.assign(score=sims)
        # Keep speeches with some relevance
        results = results[results["score"] > 0]

    # Always show newest first
    results = results.sort_values(by="date", ascending=False)

    st.write(f"Found **{len(results)}** speeches.")

    # Top quotes for query
    if query.strip():
        quotes = top_quotes(query, results, k=5)
        if quotes:
            st.subheader("Top Quotes Containing Your Keyword")
            for s, nice_date, link in quotes:
                st.markdown(f"- “{s}” — *{nice_date}*. [Speech link]({link})")

    # Results list
    for _, row in results.iterrows():
        with st.expander(f"{row['date'].strftime('%Y-%m-%d')} – {row['title']}"):
            # Themes
            themes_txt = ", ".join(row["themes_u"]) if row["themes_u"] else "N/A"
            st.markdown(f"**Themes:** {themes_txt}  ")
            st.markdown(f"**Link:** [View Speech]({row['link']})  ")

            # Bullet points (3–5 key points)
            bullets = key_sentences(row["clean_transcript"], n=4)
            if bullets:
                st.markdown("**Key Points:**")
                for b in bullets:
                    st.markdown(f"- {b}")

            # Snippet (short summary)
            st.markdown("**Snippet:** " + concise_summary(row["clean_transcript"], max_sents=2))

            # Full transcript (optional)
            with st.expander("Full Transcript (cleaned)"):
                st.write(row["clean_transcript"])

if __name__ == "__main__":
    main()
