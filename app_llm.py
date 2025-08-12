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
    format_hits_for_context, llm, llm_stream
)

_LLM_CACHE = {}


def llm_cached(key, system_prompt, user_prompt, *, model="gpt-4o-mini",
               max_tokens=700, temperature=0.3, stream=False):
    """Cache around llm helpers. When `stream=True`, content streams in real time."""
    if key in _LLM_CACHE:
        text, used_model = _LLM_CACHE[key]
        if stream:
            st.markdown(text)
        return text, used_model

    if stream:
        gen, used_model = llm_stream(system_prompt, user_prompt,
                                     model=model, max_tokens=max_tokens,
                                     temperature=temperature)
        text = st.write_stream(gen)
        out = (text, used_model)
    else:
        try:
            text = llm(system_prompt, user_prompt, model=model,
                       max_tokens=max_tokens, temperature=temperature)
            out = (text, model)
        except Exception:
            fallback = "gpt-4o-mini"
            text = llm(system_prompt, user_prompt, model=fallback,
                       max_tokens=max_tokens, temperature=temperature)
            out = (text, fallback)

    _LLM_CACHE[key] = out
    return out

# Optional Phase 3 imports (soft)
try:
    from phase3_utils import (
        build_issue_trajectory, forecast_issue_trends, trajectory_narrative
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
                st.error(f"Error: {e}")

# Load data and index
_ensure_index()
df = load_df()
index, metas, chunks = load_index()

# Query input
query = st.text_input("Search speeches", "")

# Retrieval controls
c1, c2, c3 = st.columns(3)
with c1:
    sort = st.selectbox("Sort", ["Newest", "Relevance"], index=0)
    precision_mode = st.toggle("High precision ranking", value=True)
    strictness = st.slider("Strictness", 0.0, 1.0, 0.6, 0.05)
with c2:
    exact_phrase = st.toggle("Exact phrase", value=True)
    exclude_terms = st.text_input("Exclude terms (comma-separated)", "")
    core_topic_only = st.toggle("Core topic only", value=False)
with c3:
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_from = st.date_input("Date from", min_date)
    date_to = st.date_input("Date to", max_date)
filters = {"date_from": date_from, "date_to": date_to}

view_mode = st.selectbox("View mode", ["Depth", "Breadth"], index=0)
ctx_limit = st.slider("Context items per section", 6, 20, 12)
model_preferred = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
per_item_tokens = st.slider("Tokens per item", 200, 1000, 400, step=50)

sort_key = "newest" if sort == "Newest" else "relevance"

speeches, total_before = [], 0
display_list = []
if query and query.strip():
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
    show_all = st.toggle("Show all matched speeches", value=False)
    display_list = speeches if show_all else speeches[:15]
else:
    st.info("Enter a search term above to explore speeches.")

# ---------------- Tabs ----------------
tabs = ["Results", "Thematic Evolution", "Briefing Pack", "Rapid Response", "Draft Assist"]
if _HAS_P3:
    tabs.insert(3, "Trajectory")
tab_objs = st.tabs(tabs)

# Map tabs
tab_res  = tab_objs[0]
tab_comp = tab_objs[1]
tab_brief = tab_objs[2]
if _HAS_P3:
    tab_traj = tab_objs[3]
    tab_rr  = tab_objs[4]
    tab_draft = tab_objs[5]
else:
    tab_rr   = tab_objs[3]
    tab_draft = tab_objs[4]

# Helper to convert speeches list into hits_for_llm (idx,m,ch)
def _to_hits(items):
    return [(sp["best_idx"], sp["meta"], sp["best_chunk"]) for sp in items]

# ---------------- Results tab ----------------
with tab_res:
    if not speeches:
        st.info("Enter a search above to see results.")
    else:
        st.subheader("Most Recent" if sort_key == "newest" else "Most Relevant")
        for sp in display_list:
            m = sp["meta"]; ch = sp["best_chunk"]
            st.markdown(f"**{m.get('date')} — [{m.get('title')}]({m.get('link')})**")
            sys = "You are an IMF speech analyst. Produce a concise snippet and 3–5 bullets grounded only in the excerpt."
            usr = f"Excerpt:\\n{ch}\\n\\nReturn a 1–2 sentence snippet, then 3–5 bullets of key points."
            key = f"res::{m.get('date')}::{m.get('title')}::{model_preferred}"
            (md, used_model) = llm_cached(key, sys, usr, model=model_preferred,
                                         max_tokens=per_item_tokens, temperature=0.2,
                                         stream=True)
            st.caption(f"Model: {used_model}")
        with st.expander("Show sources used on this page"):
            for (d, t, l) in sources_from_speeches(display_list):
                st.markdown(f"- {d} — [{t}]({l})")

# ---------------- Thematic Evolution (date range) ----------------
with tab_comp:
    if not speeches:
        st.info("Enter a search above to analyze thematic evolution.")
    else:
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
                if y == 0:
                    continue
                by_year.setdefault(y, []).append(sp)
            years_asc = sorted(by_year.keys())
            year_span = filters['date_to'].year - filters['date_from'].year

            ctx_for_evol = ""
            if year_span >= 2 and years_asc:
                per_year_limit = max(3, min(6, (12 if view_mode=="Depth" else 8) // max(1, len(years_asc))))
                year_sections = []
                for y in years_asc:
                    hits = _to_hits(by_year[y])[:per_year_limit]
                    ctx = format_hits_for_context(hits, limit=per_year_limit, char_limit=900)
                    if ctx.strip():
                        year_sections.append(f"=== Year {y} ===\\n{ctx}")
                full_ctx = "\\n\\n".join(year_sections) if year_sections else "(no matching context)"

                sys1 = ("You are a senior IMF communications strategist. Using ONLY the provided context, "
                        "produce issue-focused yearly summaries. Focus on substance; avoid speculation. "
                        "Add (Month YYYY — Title) from headers when referencing specifics.")
                usr1 = f"""Topic: {query}

Date range: {filters['date_from'].isoformat()} → {filters['date_to'].isoformat()}

Context grouped by year (each item starts with [YYYY-MM-DD — Title](link)):
{full_ctx}

Task: For each year in the range, list 3–5 ISSUE headings with 1–2 sentence summaries (no quotes). Keep it concise."""
                st.markdown("### Per-Year Focus")
                (focus_md, used_model1) = llm_cached(
                    f"peryear::{query}::{filters['date_from']}::{filters['date_to']}::{model_preferred}",
                    sys1, usr1, model=model_preferred, max_tokens=per_item_tokens,
                    temperature=0.2, stream=True)
                st.caption(f"Model: {used_model1}")
                ctx_for_evol = full_ctx
            else:
                ctx_range = format_hits_for_context(_to_hits(range_items), limit=(ctx_limit+4))
                sys_focus = (
                    "You are a senior IMF communications strategist. Using ONLY the provided context, "
                    "list key issue-focused takeaways for this date range."
                )
                usr_focus = f"""Topic: {query}

Date range: {filters['date_from'].isoformat()} → {filters['date_to'].isoformat()}

Context:
{ctx_range}

Task: List 3–5 ISSUE headings with 1–2 sentence summaries (no quotes)."""
                st.markdown("### Focus")
                (focus_md, used_model1) = llm_cached(
                    f"focus::{query}::{filters['date_from']}::{filters['date_to']}::{model_preferred}",
                    sys_focus, usr_focus, model=model_preferred, max_tokens=per_item_tokens,
                    temperature=0.2, stream=True)
                st.caption(f"Model: {used_model1}")
                ctx_for_evol = ctx_range

            sys2 = ("You analyze evolution across the full date range. Use ONLY context. "
                    "Be concrete: what gained emphasis, what was deemphasized, any NEW issues. "
                    "Cite with (Month YYYY — Title) based on headers.")
            usr2 = f"""Topic: {query}
Range: {filters['date_from'].isoformat()} → {filters['date_to'].isoformat()}

Context:
{ctx_for_evol}

Task: Write a short 'Messaging Evolution' narrative for the whole range (6–10 sentences, crisp)."""
            st.markdown("### Messaging Evolution")
            (evol_md, used_model2) = llm_cached(
                f"evol::{query}::{filters['date_from']}::{filters['date_to']}::{model_preferred}",
                sys2, usr2, model=model_preferred, max_tokens=per_item_tokens,
                temperature=0.2, stream=True)
            st.caption(f"Model: {used_model2}")

            with st.expander("Show sources (Thematic Evolution)"):
                for (d, t, l) in sources_from_speeches(range_items):
                    st.markdown(f"- {d} — [{t}]({l})")

# ---------------- Briefing Pack tab ----------------
with tab_brief:
    st.subheader("Briefing Pack (LLM)")
    brief_query = st.text_input("Search topic for briefing", "")
    if st.button("Generate briefing") and brief_query.strip():
        brief_speeches, _ = retrieve_speeches(
            query=brief_query,
            index=index, metas=metas, chunks=chunks,
            filters=filters,
            sort="relevance",
            precision=precision_mode,
            strictness=strictness,
            exact_phrase=exact_phrase,
            exclude_terms=[],
            core_topic_only=False,
            use_llm_rerank=True,
            model=model_preferred,
        )
        brief_hits = _to_hits(brief_speeches[: (36 if view_mode == "Depth" else 20)])
        ctx_brief = format_hits_for_context(brief_hits, limit=(ctx_limit+4))
        sys_b = "You are drafting a media-ready briefing grounded ONLY in provided context."
        usr_b = f"""Topic: {brief_query}

Context:
{ctx_brief}

Tasks:
- Executive summary (3–5 bullets).
- Five key issues (1–2 sentence summaries each; no quotes).
- Five strongest quotes (with Month YYYY — Title).
- Short timeline bullets (Month YYYY — Title).
Return clean Markdown."""
        key = f"brief::{brief_query}::{filters['date_from']}::{filters['date_to']}::{model_preferred}"
        (brief_md, used_model4) = llm_cached(key, sys_b, usr_b, model=model_preferred,
                                             max_tokens=per_item_tokens+200, temperature=0.25,
                                             stream=True)
        st.caption(f"Model: {used_model4}")
        st.download_button("Download briefing (Markdown)", brief_md.encode("utf-8"), file_name="briefing.md", mime="text/markdown")
    else:
        st.info("Enter a topic and click Generate briefing.")

# ---------------- Optional Trajectory tab ----------------
if _HAS_P3:
    with tab_traj:
        if not speeches:
            st.info("Enter a search above to view trajectory analysis.")
        else:
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

# ---------------- Rapid Response tab ----------------
with tab_rr:
    st.subheader("Rapid Response")
    inquiry = st.text_area("Media inquiry from journalist", "")
    if st.button("Generate response") and inquiry.strip():
        rr_speeches, _ = retrieve_speeches(
            query=inquiry,
            index=index, metas=metas, chunks=chunks,
            filters=filters,
            sort="relevance",
            precision=precision_mode,
            strictness=strictness,
            exact_phrase=True,
            exclude_terms[],
            core_topic_only=False,
            use_llm_rerank=True,
            model=model_preferred
        )
        rr_hits = _to_hits(rr_speeches[: (36 if view_mode == "Depth" else 20)])
        ctx_rr = format_hits_for_context(rr_hits, limit=(ctx_limit+6))
        if ctx_rr.strip():
            sys = (
                "You are Kristalina Georgieva's communications aide. "
                "Using only the provided excerpts from her speeches, answer the media inquiry "
                "with 3–5 concise bullet points followed by a short narrative paragraph."
            )
            usr = f"""Inquiry: {inquiry}

Context:
{ctx_rr}

Tasks:
- Provide 3–5 bullet points addressing the inquiry.
- Then write a short narrative paragraph synthesizing the answer."""
            key = f"rr::{hashlib.sha256((inquiry+ctx_rr).encode()).hexdigest()}::{model_preferred}"
            (rr_md, used_model) = llm_cached(key, sys, usr, model=model_preferred,
                                             max_tokens=500, temperature=0.2, stream=True)
            st.caption(f"Model: {used_model}")
            with st.expander("Show sources (Rapid Response)"):
                for (d,t,l) in sources_from_speeches(rr_speeches):
                    st.markdown(f"- {d} — [{t}]({l})")
        else:
            st.warning("No relevant context found to answer the inquiry.")

# ---------------- Draft Assist tab ----------------
with tab_draft:
    st.subheader("First Draft Speech Assist")
    big_ideas = st.text_input("Big ideas/themes (comma-separated)", "")
    r1c1, r1c2 = st.columns(2)
    audience = r1c1.text_input("Audience", "")
    venue = r1c2.text_input("Venue", "")
    r2c1, r2c2 = st.columns(2)
    tone = r2c1.text_input("Tone", "")
    style = r2c2.text_input("Style", "")

    if st.button("Draft outline") and big_ideas.strip():
        draft_speeches, _ = retrieve_speeches(
            query=big_ideas,
            index=index, metas=metas, chunks=chunks,
            filters=filters,
            sort=sort_key,
            precision=precision_mode,
            strictness=strictness,
            exact_phrase=False,
            exclude_terms[],
            core_topic_only=False,
            use_llm_rerank=True,
            model=model_preferred,
        )
        draft_hits = _to_hits(draft_speeches[: (36 if view_mode == "Depth" else 20)])
        ctx = format_hits_for_context(draft_hits, limit=(ctx_limit+6))

        params = []
        if audience.strip():
            params.append(f"Audience: {audience}")
        if venue.strip():
            params.append(f"Venue: {venue}")
        if tone.strip():
            params.append(f"Tone: {tone}")
        if style.strip():
            params.append(f"Style: {style}")
        params.append(f"Big ideas: {big_ideas}")
        param_block = "\n".join(params)

        sys = "You are drafting a first-pass speech outline in the speaker's established style, grounded ONLY in provided context."
        usr = f"""Context (date-ordered excerpts):
{ctx}

{param_block}

Tasks:
- Title and 1–2 sentence setup
- Outline with sections and 2–3 bullets each (grounded in context)
- 8–10 pull quotes with (Month YYYY — Title)
- Closing paragraph
Return clean Markdown."""
        key = f"draft::{big_ideas}::{audience}::{venue}::{tone}::{style}"
        (md, used_model) = llm_cached(key, sys, usr, model=model_preferred,
                                      max_tokens=1200, temperature=0.3, stream=True)
        st.caption(f"Model: {used_model}")
        st.download_button("Download draft (Markdown)", md.encode("utf-8"), file_name="speech_draft.md", mime="text/markdown")
    else:
        st.info("Provide big ideas and optional parameters, then click Draft outline.")
