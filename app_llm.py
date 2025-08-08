# app_llm.py — Precision Mode revamp: newest-first default, no theme filter, all results,
# but with high-precision speech-level ranking, strictness, exact-phrase/exclusions, core-topic-only,
# MMR diversification, and optional LLM reranker.
import os, datetime as _dt, hashlib
import pandas as pd
import streamlit as st
import plotly.express as px

from rag_utils import (
    load_df, load_index,
    retrieve_speeches, sources_from_speeches,
    format_hits_for_context, llm
)

# Optional Phase 2/3 imports (soft)
try:
    from phase2_utils import (
        load_playbook, load_stakeholders,
        alignment_radar_data, alignment_narrative,
        rapid_response_pack,
        tone_time_series, tone_heatmap_data,
        stakeholder_scores, stakeholder_narrative
    )
    _HAS_P2 = True
except Exception:
    _HAS_P2 = False

try:
    from phase3_utils import (
        build_issue_trajectory, forecast_issue_trends, trajectory_narrative,
        numeric_alignment_score
    )
    _HAS_P3 = True
except Exception:
    _HAS_P3 = False

st.set_page_config(page_title="Kristalina Speech Intelligence (LLM)", layout="wide")
st.title("Kristalina Georgieva — Speech Intelligence")

# ---------------- Ensure index exists ----------------
def _index_files_present():
    return all(os.path.exists(p) for p in ["index.faiss", "meta.json", "chunks.json"])

def _ensure_index():
    if _index_files_present():
        return True
    st.warning("Search index not found. Build it once in this app.")
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Set OPENAI_API_KEY in Streamlit Secrets and redeploy."); st.stop()
    if st.button("Build index now"):
        with st.spinner("Building index…"):
            try:
                from build_index import main as build_index_main
                build_index_main()
                st.success("Index built. Click **Rerun** or refresh."); st.stop()
            except Exception as e:
                st.exception(e); st.stop()
    st.stop()

_ensure_index()

# ---------------- Load data/index ----------------
@st.cache_resource
def _load_everything():
    df = load_df()
    idx, metas, chunks = load_index()
    return df, idx, metas, chunks

df, index, metas, chunks = _load_everything()

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Search")
    query = st.text_input("Topic (required)", placeholder='e.g., "climate finance" or artificial intelligence')

    # Date range (optional; defaults to full corpus)
    if len(df):
        min_d = df["date"].min().date()
        max_d = df["date"].max().date()
    else:
        min_d = max_d = _dt.date.today()

    use_range = st.checkbox("Filter by date range", value=False)
    if use_range:
        date_from = st.date_input("From", min_value=min_d, value=min_d)
        date_to   = st.date_input("To", max_value=max_d, value=max_d)
    else:
        date_from, date_to = min_d, max_d

    # Sorting (default newest)
    sort = st.radio("Sort by", ["Relevance", "Newest"], index=1, horizontal=True)

    # Precision controls
    st.subheader("Precision")
    precision_mode = st.toggle("Precision Mode", value=True, help="Speech-level ranking, thresholding, diversification, and optional LLM rerank.")
    strictness = st.slider("Strictness (on-topic threshold)", 0.4, 0.9, 0.6, 0.05)
    exact_phrase = st.checkbox("Exact phrase match (if phrase given)", value=True)
    exclude_terms = st.text_input("Exclude terms (comma-separated)", value="")
    core_topic_only = st.checkbox("Core-topic only (limit speeches to where this is a top-2 theme)", value=False,
                                  help="Uses cached LLM tagging per speech; evaluated on top candidates only.")

    # LLM mode => model/limits later
    view_mode = st.radio(
        "LLM Mode",
        ["Speed", "Depth"],
        help="Speed: gpt-4o-mini (smaller context). Depth: gpt-4o (more context).",
        horizontal=True,
        index=0
    )

# Filters only dates now
filters = {"date_from": date_from, "date_to": date_to}

# ---------------- Model choice / context limits ----------------
if view_mode == "Depth":
    model_preferred = "gpt-4o"
    ctx_limit = 14
    per_item_tokens = 950
else:
    model_preferred = "gpt-4o-mini"
    ctx_limit = 9
    per_item_tokens = 650

# ---------------- Cached LLM wrapper that returns (text, model_used) ----------------
@st.cache_data(show_spinner=False)
def llm_cached(cache_key: str, system: str, user: str, model: str, max_tokens: int, temperature: float):
    _ = hashlib.sha256((cache_key + system + user + model + str(max_tokens) + str(temperature)).encode("utf-8")).hexdigest()
    try:
        resp = llm(system, user, model=model, max_tokens=max_tokens, temperature=temperature)
        if isinstance(resp, tuple) and len(resp) == 2:
            return resp
        else:
            return (resp, model)
    except Exception:
        fb = "gpt-4o-mini" if model != "gpt-4o-mini" else "gpt-4o"
        resp = llm(system, user, model=fb, max_tokens=max_tokens, temperature=temperature)
        if isinstance(resp, tuple) and len(resp) == 2:
            return resp
        else:
            return (resp, fb)

# ---------------- Retrieval ----------------
if not query or not query.strip():
    st.info("Enter a topic to begin. Tip: use quotes for phrases, e.g., “climate finance”.")
    st.stop()

sort_key = "newest" if sort == "Newest" else "relevance"

speeches, total_before = retrieve_speeches(
    query=query,
    index=index, metas=metas, chunks=chunks,
    filters=filters,
    sort=sort_key,
    precision=precision_mode,
    strictness=strictness,
    exact_phrase=exact_phrase,
    exclude_terms=[t.strip() for t in exclude_terms.split(",") if t.strip()],
    core_topic_only=core_topic_only,
    use_llm_rerank=True,
    model=model_preferred
)

st.caption(f"Precision results: {len(speeches)} speeches (from {total_before} initial candidates).")

# Default show top 15, reveal more on click
show_all = st.toggle("Show all matched speeches", value=False)
display_list = speeches if show_all else speeches[:15]

# ---------------- Tabs ----------------
tabs = ["Results", "Thematic Evolution", "Top Quotes", "Briefing Pack", "Analytics", "Alignment", "Rapid Response", "Tone", "Draft Assist", "Stakeholders"]
if _HAS_P3:
    tabs.insert(5, "Trajectory")
tab_objs = st.tabs(tabs)

# Map tabs
tab_res   = tab_objs[0]
tab_comp  = tab_objs[1]
tab_quote = tab_objs[2]
tab_brief = tab_objs[3]
tab_viz   = tab_objs[4]
if _HAS_P3:
    tab_traj = tab_objs[5]
    tab_align = tab_objs[6]
    tab_rr    = tab_objs[7]
    tab_tone  = tab_objs[8]
    tab_draft = tab_objs[9]
    tab_stake = tab_objs[10]
else:
    tab_align = tab_objs[5]
    tab_rr    = tab_objs[6]
    tab_tone  = tab_objs[7]
    tab_draft = tab_objs[8]
    tab_stake = tab_objs[9]

# Helper to convert speeches list into hits_for_llm (idx,m,ch)
def _to_hits(items):
    return [(sp["best_idx"], sp["meta"], sp["best_chunk"]) for sp in items]

hits_for_llm = _to_hits(display_list[: (36 if view_mode == "Depth" else 20)])

# ---------------- Results tab ----------------
with tab_res:
    st.subheader("Most Recent" if sort_key=="newest" else "Most Relevant")
    for sp in display_list:
        m = sp["meta"]; ch = sp["best_chunk"]
        st.markdown(f"**{m.get('date')} — [{m.get('title')}]({m.get('link')})**")
        sys = "You are an IMF speech analyst. Produce a concise snippet and 3–5 bullets grounded only in the excerpt."
        usr = f"Excerpt:\n{ch}\n\nReturn a 1–2 sentence snippet, then 3–5 bullets of key points."
        key = f"res::{m.get('date')}::{m.get('title')}::{model_preferred}"
        (md, used_model) = llm_cached(key, sys, usr, model=model_preferred, max_tokens=per_item_tokens, temperature=0.2)
        st.markdown(md)
        st.caption(f"Model: {used_model}")
    with st.expander("Show sources used on this page"):
        for (d,t,l) in sources_from_speeches(display_list):
            st.markdown(f"- {d} — [{t}]({l})")

# ---------------- Thematic Evolution (date range) ----------------
with tab_comp:
    st.subheader("Thematic Evolution (LLM) — Date Range")
    st.caption(f"Analyzing: **{filters['date_from'].isoformat()} → {filters['date_to'].isoformat()}**")

    def _within(meta, start_date, end_date):
        import datetime as dt
        try:
            d = dt.date.fromisoformat(str(meta.get("date")))
            return start_date <= d <= end_date
        except Exception:
            return False

    range_items = [sp for sp in display_list if _within(sp["meta"], filters['date_from'], filters['date_to'])]

    if not range_items:
        st.warning("No matching content in this date range. Try broadening the range.")
    else:
        # group by year
        by_year = {}
        for sp in range_items:
            m = sp["meta"]
            y = int(m.get("year", 0) or str(m.get("date",""))[:4] or 0)
            if y == 0: continue
            by_year.setdefault(y, []).append(sp)
        years_asc = sorted(by_year.keys())

        per_year_limit = max(3, min(6, (12 if view_mode=="Depth" else 8) // max(1, len(years_asc))))
        year_sections = []
        for y in years_asc:
            hits = _to_hits(by_year[y])[:per_year_limit]
            ctx = format_hits_for_context(hits, limit=per_year_limit, char_limit=900)
            if ctx.strip():
                year_sections.append(f"=== Year {y} ===\n{ctx}")
        full_ctx = "\n\n".join(year_sections) if year_sections else "(no matching context)"

        sys1 = ("You are a senior IMF communications strategist. Using ONLY the provided context, "
                "produce issue-focused yearly summaries. Focus on substance; avoid speculation. "
                "Add (Month YYYY — Title) from headers when referencing specifics.")
        usr1 = f"""
Topic: {query}

Date range: {filters['date_from'].isoformat()} → {filters['date_to'].isoformat()}

Context grouped by year (each item starts with [YYYY-MM-DD — Title](link)):
{full_ctx}

Task: For each year in the range, list 3–5 ISSUE headings with 1–2 sentence summaries (no quotes). Keep it concise.
"""
        (per_year_md, used_model1) = llm_cached(f"peryear::{query}::{filters['date_from']}::{filters['date_to']}::{model_preferred}", sys1, usr1, model=model_preferred, max_tokens=per_item_tokens, temperature=0.2)
        st.markdown("### Per-Year Focus")
        st.markdown(per_year_md)
        st.caption(f"Model: {used_model1}")

        sys2 = ("You analyze evolution across the full date range. Use ONLY context. "
                "Be concrete: what gained emphasis, what was deemphasized, any NEW issues. "
                "Cite with (Month YYYY — Title) based on headers.")
        usr2 = f"""
Topic: {query}
Range: {filters['date_from'].isoformat()} → {filters['date_to'].isoformat()}

Use the same context as above.

Task: Write a short 'Messaging Evolution' narrative for the whole range (6–10 sentences, crisp).
"""
        (evol_md, used_model2) = llm_cached(f"evol::{query}::{filters['date_from']}::{filters['date_to']}::{model_preferred}", sys2, usr2, model=model_preferred, max_tokens=per_item_tokens, temperature=0.2)
        st.markdown("### Messaging Evolution")
        st.markdown(evol_md)
        st.caption(f"Model: {used_model2}")

        with st.expander("Show sources (Thematic Evolution)"):
            for (d, t, l) in sources_from_speeches(range_items):
                st.markdown(f"- {d} — [{t}]({l})")

# ---------------- Top Quotes tab ----------------
with tab_quote:
    st.subheader("Top Quotes (LLM)")
    ctx_quotes = format_hits_for_context(_to_hits(display_list), limit=(ctx_limit+2))
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
    (quotes_md, used_model3) = llm_cached(f"quotes::{query}::{filters['date_from']}::{filters['date_to']}::{model_preferred}", sys_q, usr_q, model=model_preferred, max_tokens=per_item_tokens, temperature=0.2)
    st.markdown(quotes_md)
    st.caption(f"Model: {used_model3}")
    with st.expander("Show sources (Quotes)"):
        for (d,t,l) in sources_from_speeches(display_list):
            st.markdown(f"- {d} — [{t}]({l})")

# ---------------- Briefing Pack tab ----------------
with tab_brief:
    st.subheader("Briefing Pack (LLM)")
    ctx_brief = format_hits_for_context(_to_hits(display_list), limit=(ctx_limit+4))
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
    (brief_md, used_model4) = llm_cached(f"brief::{query}::{filters['date_from']}::{filters['date_to']}::{model_preferred}", sys_b, usr_b, model=model_preferred, max_tokens=per_item_tokens+200, temperature=0.25)
    st.markdown(brief_md)
    st.caption(f"Model: {used_model4}")
    st.download_button("Download briefing (Markdown)", brief_md.encode("utf-8"), file_name="briefing.md", mime="text/markdown")

# ---------------- Analytics tab (visuals) ----------------
with tab_viz:
    st.subheader("Analytics & Visuals")
    viz_rows = []
    for sp in display_list:
        m = sp["meta"]
        viz_rows.append({
            "date": m.get("date"),
            "year": int(str(m.get("date"))[:4]) if m.get("date") else None,
            "title": m.get("title"),
            "link": m.get("link"),
        })
    vdf = pd.DataFrame(viz_rows)
    if not vdf.empty:
        vdf["date"] = pd.to_datetime(vdf["date"], errors="coerce")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Timeline of matching speeches")
            tl = vdf.sort_values("date")
            fig = px.scatter(tl, x="date", y=[1]*len(tl), hover_data=["title"], labels={"y":""})
            fig.update_yaxes(visible=False, showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.caption("Speeches by year (current result set)")
            year_ct = vdf.groupby("year")["title"].count().reset_index(name="speeches")
            fig2 = px.bar(year_ct.sort_values("year"), x="year", y="speeches")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data to chart yet — adjust your query or date range.")

# ---------------- Optional Trajectory tab ----------------
if _HAS_P3:
    with tab_traj:
        st.subheader("Messaging Trajectory")
        ctx = format_hits_for_context(_to_hits(display_list), limit=(ctx_limit+4))
        series = build_issue_trajectory(query, ctx, model_preferred)  # returns DataFrame columns: year, issue, share
        if not series.empty:
            fig = px.area(series, x="year", y="share", color="issue", groupnorm="fraction")
            st.plotly_chart(fig, use_container_width=True)
            forecast = forecast_issue_trends(series)
            if not forecast.empty:
                st.markdown("**Next-year trend (simple forecast)**")
                fig2 = px.bar(forecast, x="issue", y="forecast_share")
                st.plotly_chart(fig2, use_container_width=True)
            nar = trajectory_narrative(query, ctx, model_preferred)
            st.markdown(nar)
        else:
            st.info("No trajectory could be derived from current context.")

# ---------------- Alignment tab ----------------
with tab_align:
    st.subheader("Message Consistency & Alignment")
    if _HAS_P2:
        playbook = load_playbook()
        ctx = format_hits_for_context(_to_hits(display_list), limit=(ctx_limit+2))
        if _HAS_P3:
            try:
                score = numeric_alignment_score(query, playbook, ctx, model_preferred)  # 0..100
                st.metric("Alignment Score", f"{int(score)} / 100")
            except Exception:
                pass
        radar_df = alignment_radar_data(query, playbook, ctx, model_preferred)
        if radar_df is not None and not radar_df.empty and len(radar_df) >= 3:
            fig = px.line_polar(radar_df, r="score", theta="issue", line_close=True)
            fig.update_traces(fill='toself')
            st.plotly_chart(fig, use_container_width=True)
        elif radar_df is not None and not radar_df.empty:
            st.caption("Not enough distinct issues for a radar. Showing bars instead.")
            figb = px.bar(radar_df.sort_values("score", ascending=False), x="issue", y="score")
            st.plotly_chart(figb, use_container_width=True)
        else:
            st.info("No issues detected. Try broadening the range or query.")
        nar = alignment_narrative(query, playbook, ctx, model_preferred)
        st.markdown(nar)
    else:
        st.info("Playbook features are not available (phase2_utils not found).")

# ---------------- Rapid Response tab ----------------
with tab_rr:
    st.subheader("Rapid Response")
    colh1, colh2 = st.columns(2)
    headline = colh1.text_input("Paste headline or topic", "")
    url_hint = colh2.text_input("Optional: related URL (for context note only)", "")
    if st.button("Generate Press Lines"):
        ctx = format_hits_for_context(_to_hits(display_list), limit=(ctx_limit+4))
        if _HAS_P2:
            pack_md = rapid_response_pack(headline or query, ctx, model_preferred, url_hint=url_hint)
            st.markdown(pack_md)
            st.download_button("Download rapid-response (Markdown)", pack_md.encode("utf-8"), file_name="rapid_response.md", mime="text/markdown")
        else:
            st.info("Rapid response generator requires phase2_utils.")

# ---------------- Tone tab ----------------
with tab_tone:
    st.subheader("Sentiment & Tone")
    if _HAS_P2:
        ctx = format_hits_for_context(_to_hits(display_list), limit=(ctx_limit+6))
        ts = tone_time_series(query, ctx, model_preferred)
        if not ts.empty:
            fig = px.line(ts, x="date", y="score", color="tone", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        hm = tone_heatmap_data(query, ctx, model_preferred)
        if not hm.empty:
            fig2 = px.imshow(hm.pivot_table(index="tone", columns="year", values="score", fill_value=0),
                             aspect="auto", color_continuous_scale="Blues")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Tone features require phase2_utils.")

# ---------------- Draft Assist tab ----------------
with tab_draft:
    st.subheader("First Draft Speech Assist")
    a1, a2, a3 = st.columns(3)
    audience = a1.text_input("Audience", "Finance ministers")
    venue    = a2.text_input("Venue", "Annual Meetings")
    tone     = a3.selectbox("Tone", ["Balanced", "Optimistic", "Urgent", "Cautious"], index=0)
    length   = st.slider("Length (minutes)", 5, 30, 12)
    objectives = st.text_input("Top 3 objectives (comma-separated)", "Reassure markets, Highlight reforms, Call for cooperation")

    if st.button("Draft outline"):
        ctx = format_hits_for_context(_to_hits(display_list), limit=(ctx_limit+6))
        sys = "You are drafting a first-pass speech outline in the speaker's established style, grounded ONLY in provided context."
        usr = f"""
Context (date-ordered excerpts):
{ctx}

Audience: {audience}
Venue: {venue}
Tone: {tone}
Length: {length} minutes
Objectives: {objectives}

Tasks:
- Title and 1–2 sentence setup
- Outline with sections and 2–3 bullets each (grounded in context)
- 8–10 pull quotes with (Month YYYY — Title)
- Closing paragraph
Return clean Markdown.
"""
        key = f"draft::{query}::{audience}::{venue}::{tone}::{length}"
        (md, used_model) = llm_cached(key, sys, usr, model=model_preferred, max_tokens=1200, temperature=0.3)
        st.markdown(md)
        st.caption(f"Model: {used_model}")
        st.download_button("Download draft (Markdown)", md.encode("utf-8"), file_name="speech_draft.md", mime="text/markdown")

# ---------------- Stakeholders tab ----------------
with tab_stake:
    st.subheader("Stakeholder Relevance")
    if _HAS_P2:
        stakeholders = load_stakeholders()
        ctx = format_hits_for_context(_to_hits(display_list), limit=(ctx_limit+4))
        scores = stakeholder_scores(query, stakeholders, ctx, model_preferred)  # returns DataFrame
        if not scores.empty:
            fig = px.bar(scores, x="stakeholder", y="score")
            st.plotly_chart(fig, use_container_width=True)
            nar = stakeholder_narrative(query, stakeholders, ctx, model_preferred)
            st.markdown(nar)
        else:
            st.info("No clear stakeholder mapping detected—try a broader query or range.")
    else:
        st.info("Stakeholder features require phase2_utils.")
