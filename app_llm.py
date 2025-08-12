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

if not query or not query.strip():
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
            (focus_md, used_model1) = llm_cached(
                f"peryear::{query}::{filters['date_from']}::{filters['date_to']}::{model_preferred}",
                sys1, usr1, model=model_preferred, max_tokens=per_item_tokens, temperature=0.2)
            st.markdown("### Per-Year Focus")
            st.markdown(focus_md)
            st.caption(f"Model: {used_model1}")
            ctx_for_evol = full_ctx
        else:
            ctx_range = format_hits_for_context(_to_hits(range_items), limit=(ctx_limit+4))
            sys_focus = (
                "You are a senior IMF communications strategist. Using ONLY the provided context, "
                "list key issue-focused takeaways for this date range."
            )
            usr_focus = f"""
Topic: {query}

Date range: {filters['date_from'].isoformat()} → {filters['date_to'].isoformat()}

Context:
{ctx_range}

Task: List 3–5 ISSUE headings with 1–2 sentence summaries (no quotes).
"""
            (focus_md, used_model1) = llm_cached(
                f"focus::{query}::{filters['date_from']}::{filters['date_to']}::{model_preferred}",
                sys_focus, usr_focus, model=model_preferred, max_tokens=per_item_tokens, temperature=0.2)
            st.markdown("### Focus")
            st.markdown(focus_md)
            st.caption(f"Model: {used_model1}")
            ctx_for_evol = ctx_range

        sys2 = ("You analyze evolution across the full date range. Use ONLY context. "
                "Be concrete: what gained emphasis, what was deemphasized, any NEW issues. "
                "Cite with (Month YYYY — Title) based on headers.")
        usr2 = f"""
Topic: {query}
Range: {filters['date_from'].isoformat()} → {filters['date_to'].isoformat()}

Context:
{ctx_for_evol}

Task: Write a short 'Messaging Evolution' narrative for the whole range (6–10 sentences, crisp).
"""
        (evol_md, used_model2) = llm_cached(
            f"evol::{query}::{filters['date_from']}::{filters['date_to']}::{model_preferred}",
            sys2, usr2, model=model_preferred, max_tokens=per_item_tokens, temperature=0.2)
        st.markdown("### Messaging Evolution")
        st.markdown(evol_md)
        st.caption(f"Model: {used_model2}")

        with st.expander("Show sources (Thematic Evolution)"):
            for (d, t, l) in sources_from_speeches(range_items):
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
            exclude_terms=[],
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
- Then write a short narrative paragraph synthesizing the answer.
"""
            key = f"rr::{hashlib.sha256((inquiry+ctx_rr).encode()).hexdigest()}::{model_preferred}"
            (rr_md, used_model) = llm_cached(key, sys, usr, model=model_preferred, max_tokens=500, temperature=0.2)
            st.markdown(rr_md)
            st.caption(f"Model: {used_model}")
            with st.expander("Show sources (Rapid Response)"):
                for (d,t,l) in sources_from_speeches(rr_speeches):
                    st.markdown(f"- {d} — [{t}]({l})")
        else:
            st.warning("No relevant context found to answer the inquiry.")

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
