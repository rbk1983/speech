# app_llm.py — Phase 1: polished UX, controls, visuals, LLM narratives
import os, datetime as _dt
import streamlit as st
import plotly.express as px
from rag_utils import load_df, load_index, retrieve, sources_from_hits, format_hits_for_context, llm

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

# Model choice / context size
model = "gpt-4o" if view_mode == "Depth" else "gpt-4o-mini"
ctx_limit = 18 if view_mode == "Depth" else 10

# ---------- Results tab ----------
with tab_res:
    st.subheader("Most Relevant Speeches" if sort_key=="relevance" else "Most Recent Speeches")
    for (_, m, ch) in hits:
        st.markdown(f"**{m.get('date')} — [{m.get('title')}]({m.get('link')})**")
        # LLM snippet + bullets
        sys = "You are an IMF speech analyst. Produce a concise snippet and 3–5 bullets grounded only in the excerpt."
        usr = f"Excerpt:\n{ch}\n\nReturn a 1–2 sentence snippet, then 3–5 bullets of key points."
        md = llm(sys, usr, model=model, max_tokens=450, temperature=0.3)
        st.markdown(md)
    with st.expander("Show sources used on this page"):
        for (d,t,l) in sources_from_hits(hits):
            st.markdown(f"- {d} — [{t}]({l})")

# ---------- Quick Compare tab ----------
# ---------- Quick Compare tab ----------
with tab_compare:
    st.subheader("Thematic Quick Compare (LLM)")

    compare_mode = st.radio("Compare mode", ["Years", "Date ranges"], horizontal=True)

    def _context(hlist, limit=ctx_limit):
        return format_hits_for_context(hlist, limit=limit, char_limit=900)

    def _within(hit, start_date, end_date):
        try:
            d = _dt.date.fromisoformat(str(hit[1].get("date")))
            return start_date <= d <= end_date
        except Exception:
            return False

    if compare_mode == "Years":
        years_avail = sorted(df["date"].dt.year.unique())
        colA, colB = st.columns(2)
        year_a = colA.selectbox("Year A", years_avail, index=0)
        year_b = colB.selectbox("Year B", years_avail, index=min(1, len(years_avail)-1))

        # Use the CURRENT filtered result universe (query + theme + sidebar date range)
        year_a_hits = [h for h in hits if int(h[1].get("year", 0)) == int(year_a)]
        year_b_hits = [h for h in hits if int(h[1].get("year", 0)) == int(year_b)]

        label_a = f"{year_a}"
        label_b = f"{year_b}"
        ctx_a = _context(year_a_hits)
        ctx_b = _context(year_b_hits)

    else:  # Date ranges
        # Range A defaults to the sidebar global range
        colA, colB = st.columns(2)
        rngA_from = colA.date_input("Range A — From", value=date_from,
                                    min_value=df["date"].min().date(), max_value=df["date"].max().date())
        rngA_to = colA.date_input("Range A — To", value=date_to,
                                  min_value=df["date"].min().date(), max_value=df["date"].max().date())

        # Auto-suggest Range B as the previous equal-length period; user can override
        span_days = max(1, (rngA_to - rngA_from).days + 1)
        prev_to = rngA_from - _dt.timedelta(days=1)
        prev_from = prev_to - _dt.timedelta(days=span_days - 1)

        rngB_from = colB.date_input("Range B — From",
                                    value=max(df["date"].min().date(), prev_from),
                                    min_value=df["date"].min().date(), max_value=df["date"].max().date())
        rngB_to = colB.date_input("Range B — To",
                                  value=max(df["date"].min().date(), prev_to),
                                  min_value=df["date"].min().date(), max_value=df["date"].max().date())

        range_a_hits = [h for h in hits if _within(h, rngA_from, rngA_to)]
        range_b_hits = [h for h in hits if _within(h, rngB_from, rngB_to)]

        label_a = f"{rngA_from.isoformat()} → {rngA_to.isoformat()}"
        label_b = f"{rngB_from.isoformat()} → {rngB_to.isoformat()}"
        ctx_a = _context(range_a_hits)
        ctx_b = _context(range_b_hits)

    # If no content in either bucket
    if (compare_mode == "Years" and (not year_a_hits and not year_b_hits)) or \
       (compare_mode == "Date ranges" and (not range_a_hits and not range_b_hits)):
        st.warning("No matching content for these selections with the current filters. Try broadening the range or query.")
    else:
        sys = ("You are a senior IMF comms strategist. Using ONLY the provided context, write issue-focused analysis. "
               "Be concrete, policy-aware, and concise. Add (Month YYYY — Title) after points using the headers.")
        usr = f"""
Topic: {query}

Period A: {label_a}
Context:
{ctx_a}

Period B: {label_b}
Context:
{ctx_b}

Tasks:
1) For Period A: list 4–6 issue headings with 1–2 sentence summaries each (no quotes).
2) For Period B: list 4–6 issue headings with 1–2 sentence summaries each (no quotes).
3) Messaging evolution: a short narrative on what gained emphasis, what was deemphasized, and any new issues. Cite headers as (Month YYYY — Title).
"""
        cmp_md = llm(sys, usr, model=model, max_tokens=900, temperature=0.3)
        st.markdown(cmp_md)

    # Sources toggle
    with st.expander("Show sources (Quick Compare)"):
        if compare_mode == "Years":
            src_hits = year_a_hits + year_b_hits
        else:
            src_hits = range_a_hits + range_b_hits
        for (d,t,l) in sources_from_hits(src_hits):
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
