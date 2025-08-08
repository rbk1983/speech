# app_llm.py — Phase 1 (fast GPT-5, caching, polished UI, visuals)
import os, datetime as _dt, hashlib
import pandas as pd
import streamlit as st
import plotly.express as px

from rag_utils import (
    load_df, load_index, retrieve,
    sources_from_hits, format_hits_for_context, llm
)

st.set_page_config(page_title="Kristalina Speech Intelligence (LLM)", layout="wide")
st.title("Kristalina Georgieva — Speech Intelligence")

# ---------- Ensure index exists ----------
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

# ---------- Load ----------
@st.cache_resource
def _load_everything():
    df = load_df()
    idx, metas, chunks = load_index()
    return df, idx, metas, chunks

df, index, metas, chunks = _load_everything()

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Search Controls")
    query = st.text_input("Topic", placeholder="e.g., artificial intelligence, climate finance")

    # Date range
    if len(df):
        min_d = df["date"].min().date()
        max_d = df["date"].max().date()
    else:
        min_d = max_d = _dt.date.today()

    date_from = st.date_input("From", min_value=min_d, value=min_d)
    date_to = st.date_input("To", max_value=max_d, value=max_d)

    sort = st.radio("Sort by", ["Relevance", "Newest"], horizontal=True)

    # LLM mode => model/limits later
    view_mode = st.radio(
        "LLM Mode",
        ["Speed", "Depth"],
        help="Speed: GPT-5-mini with smaller context. Depth: GPT-5 with more context.",
        horizontal=True
    )

    # Pagination
    page_size = st.selectbox("Results per page", [10, 20, 30], index=1)
    if "page_offset" not in st.session_state:
        st.session_state.page_offset = 0

    st.divider()
    st.subheader("Saved Searches")
    saved = st.session_state.get("saved_searches", [])
    if st.button("⭐ Save current"):
        if query.strip():
            saved.append({
                "q": query.strip(),
                "from": str(date_from),
                "to": str(date_to),
                "sort": sort,
                "mode": view_mode
            })
            st.session_state["saved_searches"] = saved
    for i, s in enumerate(saved):
        if st.button(f"Load: {s['q']} ({s['from']}→{s['to']}, {s['sort']}, {s['mode']})"):
            query = s["q"]
            date_from = _dt.date.fromisoformat(s["from"])
            date_to = _dt.date.fromisoformat(s["to"])
            sort = s["sort"]
            view_mode = s["mode"]

# Optional theme facet
theme_col = "new_themes" if "new_themes" in df.columns else ("themes" if "themes" in df.columns else None)
all_themes = sorted({t for lst in (df[theme_col] if theme_col else []) for t in (lst if isinstance(lst, list) else [])}) if theme_col else []
theme_filter = st.multiselect("Filter by theme (optional)", all_themes, help="Click to filter results to these themes.")

filters = {"themes": theme_filter, "date_from": date_from, "date_to": date_to}

# ---------- Model choice / context limits ----------
if view_mode == "Depth":
    model = "gpt-5"        # higher quality
    ctx_limit = 12         # how many items to include in LLM context sections
    per_item_tokens = 1000
else:
    model = "gpt-5-mini"   # faster & cheaper
    ctx_limit = 8
    per_item_tokens = 700

# ---------- Cached LLM wrapper ----------
@st.cache_data(show_spinner=False)
def llm_cached(cache_key: str, system: str, user: str, model: str, max_tokens: int, temperature: float):
    # Streamlit caches on the function signature; we also derive a hash key if needed
    _ = hashlib.sha256((cache_key + system + user + model + str(max_tokens) + str(temperature)).encode("utf-8")).hexdigest()
    return llm(system, user, model=model, max_tokens=max_tokens, temperature=temperature)

# ---------- Retrieval ----------
if not query or not query.strip():
    st.info("Enter a topic in the sidebar to begin.")
    st.stop()

sort_key = "newest" if sort == "Newest" else "relevance"
limit = page_size
offset = st.session_state.page_offset

hits, total = retrieve(query, index, metas, chunks, k=100, filters=filters, sort=sort_key, offset=offset, limit=limit)
st.caption(f"Showing {len(hits)} of {total} results — page {offset//limit + 1}")

# Cap how many items feed the LLM sections (for speed)
hits_for_llm = hits[: (30 if view_mode == "Depth" else 18)]

# Pagination buttons
c1, c2, _ = st.columns([1,1,6])
with c1:
    if st.button("◀ Prev", disabled=offset==0):
        st.session_state.page_offset = max(0, offset - limit); st.rerun()
with c2:
    if st.button("Next ▶", disabled=offset + limit >= total):
        st.session_state.page_offset = offset + limit; st.rerun()

# ---------- Tabs ----------
tab_res, tab_compare, tab_quotes, tab_brief, tab_viz = st.tabs(
    ["Results", "Thematic Evolution", "Top Quotes", "Briefing Pack", "Analytics"]
)

# ---------- Results tab ----------
with tab_res:
    st.subheader("Most Relevant Speeches" if sort_key=="relevance" else "Most Recent Speeches")
    for (_, m, ch) in hits:
        st.markdown(f"**{m.get('date')} — [{m.get('title')}]({m.get('link')})**")
        sys = "You are an IMF speech analyst. Produce a concise snippet and 3–5 bullets grounded only in the excerpt."
        usr = f"Excerpt:\n{ch}\n\nReturn a 1–2 sentence snippet, then 3–5 bullets of key points."
        key = f"res::{m.get('date')}::{m.get('title')}::{model}"
        md = llm_cached(key, sys, usr, model=model, max_tokens=per_item_tokens, temperature=0.2)
        st.markdown(md)
    with st.expander("Show sources used on this page"):
        for (d,t,l) in sources_from_hits(hits):
            st.markdown(f"- {d} — [{t}]({l})")

# ---------- Thematic Evolution tab (single date range) ----------
with tab_compare:
    st.subheader("Thematic Evolution (LLM) — Date Range")
    st.caption(f"Analyzing: **{date_from.isoformat()} → {date_to.isoformat()}**")

    def _within(hit, start_date, end_date):
        try:
            d = _dt.date.fromisoformat(str(hit[1].get("date")))
            return start_date <= d <= end_date
        except Exception:
            return False

    range_hits = [h for h in hits_for_llm if _within(h, date_from, date_to)]

    if not range_hits:
        st.warning("No matching content in this date range with current filters. Try broadening the range or removing theme filters.")
    else:
        # Group by year (ascending)
        by_year = {}
        for (idx, m, ch) in range_hits:
            y = int(m.get("year", 0) or str(m.get("date",""))[:4] or 0)
            if y == 0: continue
            by_year.setdefault(y, []).append((idx, m, ch))
        years_asc = sorted(by_year.keys())

        # Smaller per-year cap to keep prompts snappy
        per_year_limit = max(3, min(6, ctx_limit // max(1, len(years_asc))))
        year_sections = []
        for y in years_asc:
            ctx = format_hits_for_context(by_year[y], limit=per_year_limit, char_limit=900)
            if ctx.strip():
                year_sections.append(f"=== Year {y} ===\n{ctx}")
        full_ctx = "\n\n".join(year_sections) if year_sections else "(no matching context)"

        # 1) Per-Year Focus
        sys1 = ("You are a senior IMF communications strategist. Using ONLY the provided context, "
                "produce issue-focused yearly summaries. Focus on substance; avoid speculation. "
                "Add (Month YYYY — Title) from headers when referencing specifics.")
        usr1 = f"""
Topic: {query}

Date range: {date_from.isoformat()} → {date_to.isoformat()}

Context grouped by year (each item starts with [YYYY-MM-DD — Title](link)):
{full_ctx}

Task: For each year in the range, list 3–5 ISSUE headings with 1–2 sentence summaries (no quotes). Keep it concise.
"""
        per_year_md = llm_cached(f"peryear::{query}::{date_from}::{date_to}::{model}", sys1, usr1, model=model, max_tokens=per_item_tokens, temperature=0.2)
        st.markdown("### Per-Year Focus")
        st.markdown(per_year_md)

        # 2) Evolution Narrative
        sys2 = ("You analyze evolution across the full date range. Use ONLY context. "
                "Be concrete: what gained emphasis, what was deemphasized, any NEW issues. "
                "Cite with (Month YYYY — Title) based on headers.")
        usr2 = f"""
Topic: {query}
Range: {date_from.isoformat()} → {date_to.isoformat()}

Use the same context as above.

Task: Write a short 'Messaging Evolution' narrative for the whole range (6–10 sentences, crisp).
"""
        evol_md = llm_cached(f"evol::{query}::{date_from}::{date_to}::{model}", sys2, usr2, model=model, max_tokens=per_item_tokens, temperature=0.2)
        st.markdown("### Messaging Evolution")
        st.markdown(evol_md)

        with st.expander("Show sources (Thematic Evolution)"):
            for (d, t, l) in sources_from_hits(range_hits):
                st.markdown(f"- {d} — [{t}]({l})")

# ---------- Top Quotes tab ----------
with tab_quotes:
    st.subheader("Top Quotes (LLM)")
    ctx_quotes = format_hits_for_context(hits_for_llm, limit=ctx_limit+2)
    sys_q = ("You extract on-topic quotes. Use ONLY provided context; return exact sentences. "
             "Each bullet ends with (Month YYYY — Title) and includes the link from the header. "
             "Prioritize quotes that clearly address the user's topic.")
    usr_q = f"""
Topic: {query}

Context:
{ctx_quotes}

Task:
Return exactly 5 strong on-topic quotes (1–3 sentences each). Each bullet ends with (Month YYYY — Title).
"""
    quotes_md = llm_cached(f"quotes::{query}::{date_from}::{date_to}::{model}", sys_q, usr_q, model=model, max_tokens=per_item_tokens, temperature=0.2)
    st.markdown(quotes_md)
    with st.expander("Show sources (Quotes)"):
        for (d,t,l) in sources_from_hits(hits_for_llm):
            st.markdown(f"- {d} — [{t}]({l})")

# ---------- Briefing Pack tab ----------
with tab_brief:
    st.subheader("Briefing Pack (LLM)")
    ctx_brief = format_hits_for_context(hits_for_llm, limit=ctx_limit+4)
    sys_b = "You are drafting a media-ready briefing grounded ONLY in provided context."
    usr_b = f"""
Topic: {query}

Context:
{ctx_brief}

Tasks:
- Executive summary (3–5 bullets).
- Five key issues (1–2 sentence summaries each; no quotes).
- Five strongest quotes (with Month YYYY — Title).
- Short timeline bullets (Month YYYY — Title).
Return clean Markdown.
"""
    brief_md = llm_cached(f"brief::{query}::{date_from}::{date_to}::{model}", sys_b, usr_b, model=model, max_tokens=per_item_tokens+200, temperature=0.25)
    st.markdown(brief_md)
    st.download_button("Download briefing (Markdown)", brief_md.encode("utf-8"), file_name="briefing.md", mime="text/markdown")

# ---------- Analytics tab (visuals) ----------
with tab_viz:
    st.subheader("Analytics & Visuals")

    # Build a tiny frame of returned results for charts
    viz_rows = []
    for (_, m, _ch) in hits:
        viz_rows.append({
            "date": m.get("date"),
            "year": int(str(m.get("date"))[:4]) if m.get("date") else None,
            "title": m.get("title"),
            "link": m.get("link"),
            "themes": ", ".join(m.get("themes") or [])
        })
    vdf = pd.DataFrame(viz_rows)
    if not vdf.empty:
        vdf["date"] = pd.to_datetime(vdf["date"], errors="coerce")

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Timeline of matching speeches")
            tl = vdf.sort_values("date")
            fig = px.scatter(tl, x="date", y=[1]*len(tl), hover_data=["title","themes"], labels={"y":""})
            fig.update_yaxes(visible=False, showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.caption("Speeches by year (current result set)")
            year_ct = vdf.groupby("year")["title"].count().reset_index(name="speeches")
            fig2 = px.bar(year_ct.sort_values("year"), x="year", y="speeches")
            st.plotly_chart(fig2, use_container_width=True)

        if theme_col:
            st.caption("Top themes in current results")
            explode = []
            for (_, m, _ch) in hits:
                for t in (m.get("themes") or []):
                    explode.append({"theme": t})
            tdf = pd.DataFrame(explode)
            if not tdf.empty:
                top_t = tdf["theme"].value_counts().reset_index()
                top_t.columns = ["theme","count"]
                fig3 = px.bar(top_t.head(15), x="theme", y="count")
                st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No data to chart yet — adjust your query or date range.")
