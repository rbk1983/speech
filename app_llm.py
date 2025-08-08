# NOTE: This is a full app_llm.py including Rapid Response news integration with sub-queries + relevance table.
# If you already run Phase 2 app, you can replace your app_llm.py with this version.
import os, datetime as _dt, hashlib
import pandas as pd
import streamlit as st
import plotly.express as px

from rag_utils import (
    load_df, load_index, retrieve,
    sources_from_hits, format_hits_for_context, llm
)
from phase2_utils import (
    load_playbook, load_stakeholders,
    alignment_radar_data, alignment_narrative,
    rapid_response_pack,
    tone_time_series, tone_heatmap_data,
    stakeholder_scores, stakeholder_narrative
)
from news_utils import fetch_news  # <-- new

st.set_page_config(page_title="Kristalina Speech Intelligence (LLM)", layout="wide")
st.title("Kristalina Georgieva — Speech Intelligence")

def _index_files_present():
    return all(os.path.exists(p) for p in ["index.faiss", "meta.json", "chunks.json"])

def _ensure_index():
    if _index_files_present():
        return True
    st.warning("Search index not found here. Build it once in this app.")
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Set OPENAI_API_KEY in Streamlit Secrets and redeploy.")
        st.stop()
    if st.button("Build index now"):
        with st.spinner("Building index…"):
            try:
                from build_index import main as build_index_main
                build_index_main()
                st.success("Index built. Click **Rerun** or refresh.")
                st.stop()
            except Exception as e:
                st.exception(e); st.stop()
    st.stop()

_ensure_index()

@st.cache_resource
def _load_everything():
    df = load_df()
    idx, metas, chunks = load_index()
    return df, idx, metas, chunks

df, index, metas, chunks = _load_everything()

with st.sidebar:
    st.header("Search Controls")
    query = st.text_input("Topic", placeholder='e.g., "artificial intelligence" economics, climate finance')
    if len(df):
        min_d = df["date"].min().date()
        max_d = df["date"].max().date()
    else:
        min_d = max_d = _dt.date.today()
    date_from = st.date_input("From", min_value=min_d, value=min_d)
    date_to   = st.date_input("To", max_value=max_d, value=max_d)
    sort = st.radio("Sort by", ["Relevance", "Newest"], horizontal=True)
    view_mode = st.radio("LLM Mode", ["Speed","Depth"], horizontal=True)
    page_size = st.selectbox("Results per page", [10,20,30], index=1)
    if "page_offset" not in st.session_state:
        st.session_state.page_offset = 0

theme_col = "new_themes" if "new_themes" in df.columns else ("themes" if "themes" in df.columns else None)
all_themes = sorted({t for lst in (df[theme_col] if theme_col else []) for t in (lst if isinstance(lst, list) else [])}) if theme_col else []
theme_filter = st.multiselect("Filter by theme (optional)", all_themes)

filters = {"themes": theme_filter, "date_from": date_from, "date_to": date_to}

# Model + limits
if view_mode == "Depth":
    model_preferred = "gpt-4o"; ctx_limit=12; per_item_tokens=950
else:
    model_preferred = "gpt-4o-mini"; ctx_limit=8; per_item_tokens=650

@st.cache_data(show_spinner=False)
def llm_cached(cache_key: str, system: str, user: str, model: str, max_tokens: int, temperature: float):
    _ = hashlib.sha256((cache_key + system + user + model + str(max_tokens) + str(temperature)).encode("utf-8")).hexdigest()
    try:
        resp = llm(system, user, model=model, max_tokens=max_tokens, temperature=temperature)
        if isinstance(resp, tuple) and len(resp)==2: return resp
        return (resp, model)
    except Exception:
        fb = "gpt-4o-mini" if model!="gpt-4o-mini" else "gpt-4o"
        resp = llm(system, user, model=fb, max_tokens=max_tokens, temperature=temperature)
        if isinstance(resp, tuple) and len(resp)==2: return resp
        return (resp, fb)

if not query or not query.strip():
    st.info("Enter a topic in the sidebar to begin."); st.stop()

sort_key = "newest" if sort=="Newest" else "relevance"
limit = page_size
offset = st.session_state.page_offset

hits, total = retrieve(query, index, metas, chunks, k=100, filters=filters, sort=sort_key, offset=offset, limit=limit)
st.caption(f"Showing {len(hits)} of {total} results — page {offset//limit + 1}")
hits_for_llm = hits[: (28 if view_mode=="Depth" else 16)]

c1,c2,_=st.columns([1,1,6])
with c1:
    if st.button("◀ Prev", disabled=offset==0):
        st.session_state.page_offset = max(0, offset-limit); st.rerun()
with c2:
    if st.button("Next ▶", disabled=offset+limit>=total):
        st.session_state.page_offset = offset+limit; st.rerun()

tab_res, tab_compare, tab_quotes, tab_brief, tab_viz, tab_align, tab_rr = st.tabs(
    ["Results","Thematic Evolution","Top Quotes","Briefing Pack","Analytics","Alignment","Rapid Response"]
)

# --- Results tab (omitted here for brevity in this patch) ---
with tab_res:
    st.subheader("Results")
    for (_, m, ch) in hits:
        st.markdown(f"**{m.get('date')} — [{m.get('title')}]({m.get('link')})**")
        sys="You are an IMF speech analyst. Produce a concise snippet and 3–5 bullets grounded only in the excerpt."
        usr=f"Excerpt:\n{ch}\n\nReturn a 1–2 sentence snippet, then 3–5 bullets of key points."
        key=f"res::{m.get('date')}::{m.get('title')}::{model_preferred}"
        md, used = llm_cached(key, sys, usr, model=model_preferred, max_tokens=per_item_tokens, temperature=0.2)
        st.markdown(md); st.caption(f"Model: {used}")

# --- Rapid Response tab (with live news & subquery display) ---
with tab_rr:
    st.subheader("Rapid Response — Live News")
    colh1, colh2, colh3 = st.columns([2,1,1])
    headline = colh1.text_input("Use this topic/headline for generation (defaults to sidebar Topic)", value=query)
    lookback = colh2.selectbox("Look back", ["6h","12h","24h","48h","72h"], index=2)
    show_details = colh3.checkbox("Show sub-queries & relevance table", value=True)

    if st.button("Fetch live news"):
        hours = int(lookback.replace("h",""))
        with st.spinner("Fetching Google News…"):
            results, terms = fetch_news(query, since_hours=hours, limit=30, return_terms=True)
        st.session_state["rr_results"] = results
        st.session_state["rr_terms"] = terms

    results = st.session_state.get("rr_results", [])
    terms = st.session_state.get("rr_terms", [])

    if show_details and terms:
        st.caption("Sub-queries used: " + " · ".join([f"`{t}`" for t in terms]))

    if results:
        # Build table
        df_news = pd.DataFrame([{
            "✓": False,
            "score": round(float(it.get("score",0)),3),
            "published_at": it.get("published_at",""),
            "source": it.get("source",""),
            "title": it.get("title",""),
            "url": it.get("url",""),
        } for it in results])
        st.caption("Top headlines (ranked): check the rows you want to use for press lines")
        edited = st.data_editor(df_news, num_rows="fixed", use_container_width=True, hide_index=True)
        # Collect selected
        selected = edited[edited["✓"]==True].to_dict(orient="records")

        if st.button("Generate press lines from selected"):
            if not selected:
                st.warning("Select at least one headline first.")
            else:
                # Build context
                ctx = format_hits_for_context(hits_for_llm, limit=(ctx_limit+4))
                # Compose a compact news block
                news_block = "\n".join([f"- {r['title']} ({r['source']}, {r['published_at']}) — {r['url']}" for r in selected])
                sys = ("You write rapid-response lines consistent with historic remarks. "
                       "Use ONLY the provided speech context and selected headlines. "
                       "Tone: calm, factual, forward-looking, no speculation.")
                usr = f"""Topic/headline: {headline or query}

Speech context excerpts:
{ctx}

Selected headlines:
{news_block}

Tasks:
- 3–4 press lines (bulleted), one sentence each, aligned with prior remarks.
- 3 policy specifics (bulleted) grounded in context with (Month YYYY — Title).
- 3 suggested reporter Q&As (Q and a short A anchored in context).
Return clean Markdown.
"""
                key=f"rr_live::{query}::{lookback}::{len(selected)}"
                md, used = llm_cached(key, sys, usr, model=model_preferred, max_tokens=900, temperature=0.2)
                st.markdown(md); st.caption(f"Model: {used}")
                st.download_button("Download rapid-response (Markdown)", md.encode("utf-8"),
                                   file_name="rapid_response.md", mime="text/markdown")
    else:
        st.info("Click **Fetch live news** to pull the latest headlines for your topic.")
