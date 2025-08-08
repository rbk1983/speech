# app_llm.py — Phase 1: polished UX, controls, visuals, LLM narratives
import os, datetime as _dt
import streamlit as st
import plotly.express as px
import pandas as pd
import hashlib  # ← add this
from rag_utils import load_df, load_index, retrieve, sources_from_hits, format_hits_for_context, llm

@st.cache_data(show_spinner=False)
def llm_cached(key: str, system: str, user: str, model: str, max_tokens: int, temperature: float):
    # Simple cache key: hash the inputs
    cache_key = hashlib.sha256((key + system + user + model + str(max_tokens) + str(temperature)).encode("utf-8")).hexdigest()
    # Streamlit caches on function args; we also return the text so repeats are instant
    text = llm(system, user, model=model, max_tokens=max_tokens, temperature=temperature)
    return text


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
    years = sorted(df["date"].dt.year.unique().tolist())
    if years:
        date_from = st.date_input("From", min_value=df["date"].min().date(),
                                  value=df["date"].min().date())
        date_to = st.date_input("To", max_value=df["date"].max().date(),
                                value=df["date"].max().date())
    else:
        date_from = date_to = None

    sort = st.radio("Sort by", ["Relevance", "Newest"], horizontal=True)
    view_mode = st.radio("LLM Mode", ["Speed", "Depth"], help="Speed: faster/cheaper. Depth: more context & higher-quality model.", horizontal=True)

    # Pagination
    page_size = st.selectbox("Results per page", [10, 20, 30], index=1)
    if "page_offset" not in st.session_state:
        st.session_state.page_offset = 0

    # Saved searches (session)
    st.divider()
    st.subheader("Saved Searches")
    saved = st.session_state.get("saved_searches", [])
    if st.button("⭐ Save current"):
        if query.strip():
            saved.append({"q": query.strip(), "from": str(date_from), "to": str(date_to), "sort": sort})
            st.session_state["saved_searches"] = saved
    for i, s in enumerate(saved):
        if st.button(f"Load: {s['q']} ({s['from']}→{s['to']}, {s['sort']})"):
            query = s["q"]
            date_from = _dt.date.fromisoformat(s["from"])
            date_to = _dt.date.fromisoformat(s["to"])
            sort = s["sort"]

# Optional theme facet
theme_col = "new_themes" if "new_themes" in df.columns else ("themes" if "themes" in df.columns else None)
all_themes = sorted({t for lst in (df[theme_col] if theme_col else []) for t in (lst if isinstance(lst, list) else [])}) if theme_col else []
theme_filter = st.multiselect("Filter by theme (optional)", all_themes, help="Click to filter results to these themes.")

filters = {
    "themes": theme_filter,
    "date_from": date_from,
    "date_to": date_to
}

# ---------- Run retrieval ----------
if not query or not query.strip():
    st.info("Enter a topic in the sidebar to begin.")
    st.stop()

sort_key = "newest" if sort == "Newest" else "relevance"
limit = page_size
offset = st.session_state.page_offset

hits, total = retrieve(query, index, metas, chunks, k=100, filters=filters, sort=sort_key, offset=offset, limit=limit)
st.caption(f"Showing {len(hits)} of {total} results — page {offset//limit + 1}")

# Pagination buttons
c1, c2, c3 = st.columns([1,1,6])
with c1:
    if st.button("◀ Prev", disabled=offset==0):
        st.session_state.page_offset = max(0, offset - limit)
        st.rerun()
with c2:
    if st.button("Next ▶", disabled=offset + limit >= total):
        st.session_state.page_offset = offset + limit
        st.rerun()

# ---------- Tabs ----------
tab_res, tab_compare, tab_quotes, tab_brief, tab_viz = st.tabs(
    ["Results", "Quick Compare", "Top Quotes", "Briefing Pack", "Analytics"]
)

# Model choice / context size (tighter limits for speed)
if view_mode == "Depth":
    model = "gpt-5"        # higher quality, slower
    ctx_limit = 12         # fewer chunks than before to keep latency reasonable
    per_item_tokens = 1000 # generous but not huge
else:
    model = "gpt-5-mini"   # fast & cheap
    ctx_limit = 8          # trim context to speed up
    per_item_tokens = 700


# ---------- Results tab ----------
for (_, m, ch) in hits:
    st.markdown(f"**{m.get('date')} — [{m.get('title')}]({m.get('link')})**")
    sys = "You are an IMF speech analyst. Produce a concise snippet and 3–5 bullets grounded only in the excerpt."
    usr = f"Excerpt:\n{ch}\n\nReturn a 1–2 sentence snippet, then 3–5 bullets of key points."
    # cache key based on content + model
    key = f"res::{m.get('date')}::{m.get('title')}::{model}"
    md = llm_cached(key, sys, usr, model=model, max_tokens=per_item_tokens, temperature=0.2)
    st.markdown(md)


# ---------- Thematic Evolution tab (single date range) ----------
with tab_compare:
    st.subheader("Thematic Evolution (LLM) — Date Range")

    # We use the global sidebar date range: date_from, date_to
    st.caption(f"Analyzing: **{date_from.isoformat()} → {date_to.isoformat()}**")

    def _within(hit, start_date, end_date):
        try:
            d = _dt.date.fromisoformat(str(hit[1].get("date")))
            return start_date <= d <= end_date
        except Exception:
            return False

    # Use the current filtered result universe (query + theme + sidebar range)
    range_hits = [h for h in hits if _within(h, date_from, date_to)]

    if not range_hits:
        st.warning("No matching content in this date range with current filters. Try broadening the range or removing theme filters.")
    else:
        # Group retrieved hits by year (ascending)
        by_year = {}
        for (idx, m, ch) in range_hits:
            y = int(m.get("year", 0) or str(m.get("date",""))[:4] or 0)
            if y == 0: 
                continue
            by_year.setdefault(y, []).append((idx, m, ch))
        years_asc = sorted(by_year.keys())

        # Build a structured context: "Year YYYY" section with several excerpts
        # Use a modest limit per year so we don't overload the model
        per_year_limit = max(3, min(6, ctx_limit // max(1, len(years_asc))))
        year_sections = []
        for y in years_asc:
            ctx = format_hits_for_context(by_year[y], limit=per_year_limit, char_limit=900)
            if ctx.strip():
                year_sections.append(f"=== Year {y} ===\n{ctx}")
        full_ctx = "\n\n".join(year_sections) if year_sections else "(no matching context)"

        # LLM prompt: issues per year + evolution narrative across the whole span
        system = (
            "You are a senior IMF communications strategist. Using ONLY the provided context, "
            "analyze how messaging evolved across the date range. Focus on substance (policies, instruments, risks, rationales), "
            "avoid speculation, and keep it concise and media-ready. When referencing specifics, add (Month YYYY — Title) "
            "based on the headers."
        )
        user = f"""
Topic: {query}

Date range: {date_from.isoformat()} → {date_to.isoformat()}

Context grouped by year (each item starts with [YYYY-MM-DD — Title](link)):
{full_ctx}

Tasks:
1) For each year in the range, list 4–6 ISSUE headings with 1–2 sentence summaries each (no quotes). Use clear, non-overlapping issues.
2) Messaging evolution (range-wide): a short, concrete narrative of what gained emphasis, what was deemphasized, and any NEW issues that emerged.
3) (Optional) Mini shift timeline (3–6 bullets): Month YYYY — Title — one clause on the nature of the shift.

Return clean Markdown with sections: "Per-Year Focus", "Messaging Evolution", and "Shift Timeline" (only include the last if warranted).
"""
        cmp_md = llm(system, user, model=model, max_tokens=1100, temperature=0.3)
        st.markdown(cmp_md)

        with st.expander("Show sources (Thematic Evolution)"):
            for (d, t, l) in sources_from_hits(range_hits):
                st.markdown(f"- {d} — [{t}]({l})")



# ---------- Top Quotes tab ----------
with tab_quotes:
    st.subheader("Top Quotes (LLM)")
    # Ensure we pass enough context but dedup speeches
    sys = ("You extract on-topic quotes. Use ONLY provided context; return exact sentences. "
           "Each bullet ends with (Month YYYY — Title) and includes the link from the header. "
           "Prioritize quotes that clearly address the user's topic.")
    usr = f"""
Topic: {query}

Context:
{format_hits_for_context(hits, limit=ctx_limit+4)}

Task:
Return exactly 5 strong on-topic quotes (1–3 sentences each). Each bullet ends with (Month YYYY — Title).
"""
    quotes_md = llm(sys, usr, model=model, max_tokens=700, temperature=0.2)
    st.markdown(quotes_md)
    with st.expander("Show sources (Quotes)"):
        for (d,t,l) in sources_from_hits(hits):
            st.markdown(f"- {d} — [{t}]({l})")

# ---------- Briefing Pack tab ----------
with tab_brief:
    st.subheader("Briefing Pack (LLM)")
    sys = "You are drafting a media-ready briefing grounded ONLY in provided context."
    usr = f"""
Topic: {query}

Context:
{format_hits_for_context(hits, limit=ctx_limit+6)}

Tasks:
- Executive summary (3–5 bullets).
- Five key issues (1–2 sentence summaries each; no quotes).
- Five strongest quotes (with Month YYYY — Title).
- Short timeline bullets (Month YYYY — Title).
Return clean Markdown.
"""
    brief_md = llm(sys, usr, model=model, max_tokens=1200, temperature=0.3)
    st.markdown(brief_md)

    # Simple download as Markdown
    st.download_button("Download briefing (Markdown)", brief_md.encode("utf-8"), file_name="briefing.md", mime="text/markdown")

# ---------- Analytics tab (visuals) ----------
with tab_viz:
    st.subheader("Analytics & Visuals")

    # Build a tiny frame of *returned* results for charts (not entire corpus)
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
            # explode themes for counting
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
