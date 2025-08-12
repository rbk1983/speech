# app_llm.py — Simplified UI, real newest-first, pagination
import os, datetime as _dt, hashlib, math
import pandas as pd
import streamlit as st
import plotly.express as px

# --- Utilities with safe fallback for `llm_stream` ---
from rag_utils import (
    load_df, load_index,
    retrieve_speeches, sources_from_speeches,
    format_hits_for_context, llm
)

try:
    from rag_utils import llm_stream  # type: ignore
except Exception:
    def llm_stream(system_prompt, user_prompt, *, model="gpt-4o-mini",
                   max_tokens=700, temperature=0.3):
        text = llm(system_prompt, user_prompt,
                   model=model, max_tokens=max_tokens, temperature=temperature)
        def _gen():
            yield text
        return _gen(), model

# ----- Defaults (kept simple) -----
PER_ITEM_TOKENS = 400   # Longer = richer bullets/snippets (more cost); Shorter = terser (cheaper)
DEFAULT_MODEL   = "gpt-4o-mini"

_LLM_CACHE = {}
def llm_cached(key, system_prompt, user_prompt, *, model=DEFAULT_MODEL,
               max_tokens=PER_ITEM_TOKENS, temperature=0.3, stream=False):
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
            fallback = DEFAULT_MODEL
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

# ---------------- Styling (Sticky top bar) ----------------
st.markdown("""
<style>
.sticky-bar{
  position: sticky; top: 0; z-index: 999;
  padding: .5rem .25rem .6rem .25rem;
  background: white;
  border-bottom: 1px solid rgba(0,0,0,.08);
}
.card{
  border: 1px solid rgba(0,0,0,.08);
  padding: .75rem 1rem; border-radius: 12px; margin-bottom: .75rem;
  background: #ffffff;
  box-shadow: 0 1px 4px rgba(0,0,0,.04);
}
.card h4{ margin: 0 0 .25rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("Kristalina Georgieva — Speech Intelligence")

# ---------------- Robust loaders with clearer errors ----------------
def _index_files_present():
    return all(os.path.exists(p) for p in ["index.faiss", "meta.json", "chunks.json"])

def _ensure_index():
    if _index_files_present():
        return True
    st.warning("Search index not found. You can build it here once.")
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Missing OpenAI API key. Add it to Streamlit secrets, then rerun."); st.stop()
    if st.button("Build index now"):
        with st.spinner("Building index…"):
            try:
                from build_index import main as build_index_main
                build_index_main()
                st.success("Index built. Click **Rerun** or refresh."); st.stop()
            except Exception as e:
                st.error(f"Index build failed: {e}")
                st.stop()

_ensure_index()

try:
    df = load_df()
except Exception as e:
    st.error(f"Failed to load metadata dataframe: {e}")
    st.stop()

try:
    index, metas, chunks = load_index()
except Exception as e:
    st.error(f"Failed to load search index: {e}")
    st.stop()

# ---------------- Sticky Top Bar ----------------
with st.container():
    st.markdown('<div class="sticky-bar">', unsafe_allow_html=True)
    top_c1, top_c2, top_c3, top_c4 = st.columns([2.2, 2.2, 1.2, 1.2])
    with top_c1:
        query = st.text_input("Search speeches", "", placeholder="e.g., climate finance, inflation, Ukraine")
    with top_c2:
        # Date range
        try:
            min_date = pd.to_datetime(df["date"], errors="coerce").dropna().min().date()
            max_date = pd.to_datetime(df["date"], errors="coerce").dropna().max().date()
        except Exception:
            min_date = _dt.date(2000,1,1)
            max_date = _dt.date.today()
        date_from = st.date_input("From", min_date, key="from")
        date_to = st.date_input("To", max_date, key="to")
    with top_c3:
        sort = st.selectbox("Sort", ["Newest", "Relevance"], index=0)
        layout_mode = st.selectbox("Layout", ["Dense list", "Compact cards"], index=1)
    with top_c4:
        exact_phrase = st.toggle("Exact phrase", value=True, help="Match the exact phrase; reduces noise for multi-word queries.")
        high_precision = st.toggle("High precision ranking", value=True, help="Better ranking via dense retrieval + reranking + diversity controls.")
    st.markdown('</div>', unsafe_allow_html=True)

# Simplification note: We keep DEPTH as the default philosophy (richer context per item).
st.caption("Mode: **Depth** — returns fewer items per step but with richer context for better summaries.")

# Advanced (collapsed)
with st.expander("Advanced settings", expanded=False):
    strictness = st.slider("Strictness", 0.0, 1.0, 0.6, 0.05,
                           help="Higher = tighter match to your query; lower = looser, more exploratory.")
    core_topic_only = st.toggle("Core topic only", value=False,
                                help="Show speeches where the query is a central theme (not just a passing mention).")
    model_preferred = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)

filters = {"date_from": date_from, "date_to": date_to}
sort_key = "newest" if sort == "Newest" else "relevance"

def _parse_date(meta_date):
    try:
        return pd.to_datetime(str(meta_date), errors="coerce")
    except Exception:
        return pd.NaT

def _enforce_newest_first(items):
    return sorted(items, key=lambda sp: _parse_date(sp["meta"].get("date")), reverse=True)

# Session state for pagination
if "page" not in st.session_state:
    st.session_state.page = 1

def reset_pagination():
    st.session_state.page = 1

speeches, total_before = [], 0
if query and query.strip():
    # Reset pagination when query/filters change (simple heuristic)
    # Use a fingerprint to detect changes
    fp = f"{query}|{date_from}|{date_to}|{sort_key}|{exact_phrase}|{high_precision}|{core_topic_only}|{strictness}"
    if st.session_state.get("fp") != fp:
        st.session_state.fp = fp
        reset_pagination()
    try:
        speeches, total_before = retrieve_speeches(
            query=query,
            index=index, metas=metas, chunks=chunks,
            filters=filters,
            sort=sort_key,
            precision=high_precision,
            strictness=strictness,
            exact_phrase=exact_phrase,
            exclude_terms=[],            # removed from UI by request
            core_topic_only=core_topic_only,
            use_llm_rerank=True,
            model=model_preferred
        )
    except Exception as e:
        st.error(f"Search failed: {e}")
        speeches = []

    if speeches:
        if sort_key == "newest":
            speeches = _enforce_newest_first(speeches)

        st.caption(f"Matched speeches: {len(speeches)} (from {total_before} initial candidates).")
    else:
        st.info("No results. Try loosening strictness or widening the date range.")

else:
    st.info("Enter a search term above to explore speeches.")

# ---------------- Tabs ----------------
tabs = ["Results", "Thematic Evolution", "Briefing Pack", "Rapid Response", "Draft Assist"]
try:
    if _HAS_P3:
        tabs.insert(3, "Trajectory")
except Exception:
    pass
tab_objs = st.tabs(tabs)

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

def _to_hits(items):
    return [(sp["best_idx"], sp["meta"], sp["best_chunk"]) for sp in items]

# ---------------- Results with pagination (10 per page) ----------------
with tab_res:
    if not speeches:
        st.info("Enter a search to see results.")
    else:
        st.subheader("Most Recent" if sort_key == "newest" else "Most Relevant")

        PAGE_SIZE = 10
        total = len(speeches)
        total_pages = max(1, math.ceil(total / PAGE_SIZE))
        page = max(1, min(st.session_state.page, total_pages))

        start = (page - 1) * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)
        page_items = speeches[start:end]

        # Render page items
        for sp in page_items:
            m = sp["meta"]; ch = sp["best_chunk"]
            date_s = str(m.get('date'))
            title = m.get('title')
            link = m.get('link')
            if layout_mode == "Compact cards":
                st.markdown(f'<div class="card"><h4>{date_s} — <a href="{link}">{title}</a></h4>', unsafe_allow_html=True)
                sys = "You are an IMF speech analyst. Produce a concise snippet and 3–5 bullets grounded only in the excerpt."
                usr = f"Excerpt:\n{ch}\n\nReturn a 1–2 sentence snippet, then 3–5 bullets of key points."
                key = f"res::{date_s}::{title}::{model_preferred}"
                (md, used_model) = llm_cached(key, sys, usr, model=model_preferred,
                                              max_tokens=PER_ITEM_TOKENS, temperature=0.2,
                                              stream=True)
                st.caption(f"Model: {used_model}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"**{date_s} — [{title}]({link})**")
                sys = "You are an IMF speech analyst. Produce a concise snippet and 3–5 bullets grounded only in the excerpt."
                usr = f"Excerpt:\n{ch}\n\nReturn a 1–2 sentence snippet, then 3–5 bullets of key points."
                key = f"res::{date_s}::{title}::{model_preferred}"
                (md, used_model) = llm_cached(key, sys, usr, model=model_preferred,
                                              max_tokens=PER_ITEM_TOKENS, temperature=0.2,
                                              stream=True)
                st.caption(f"Model: {used_model}")

        # Pagination controls
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_a:
            if st.button("◀︎ Prev", disabled=(page <= 1)):
                st.session_state.page = max(1, page - 1)
                st.experimental_rerun()
        with col_b:
            st.markdown(f"<div style='text-align:center'>Page {page} of {total_pages} • Showing {start+1}–{end} of {total}</div>", unsafe_allow_html=True)
        with col_c:
            if st.button("Next ▶︎", disabled=(page >= total_pages)):
                st.session_state.page = min(total_pages, page + 1)
                st.experimental_rerun()

        with st.expander("Sources used on this page"):
            page_sources = list(sources_from_speeches(page_items))
            if not page_sources:
                st.info("No sources to show.")
            else:
                for (d, t, l) in page_sources:
                    st.markdown(f"- {d} — [{t}]({l})")
                lines = [f"{d} — {t} — {l}" for (d,t,l) in page_sources]
                all_sources_text = "\n".join(lines)
                st.markdown("**Copy all sources:**")
                st.text_area("All sources (copy)", all_sources_text, height=120, label_visibility="collapsed")
                st.download_button("Download sources (.txt)", all_sources_text.encode("utf-8"),
                                   file_name="sources.txt", mime="text/plain")

# ---------------- Thematic Evolution (Depth default) ----------------
with tab_comp:
    if not speeches:
        st.info("Enter a search to analyze thematic evolution.")
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

        # Use the items currently visible (page) for a focused, depth-first feel
        range_items = [sp for sp in page_items if _within(sp["meta"], filters['date_from'], filters['date_to'])]

        if not range_items:
            st.warning("No matching content on this page in this date range. Try moving pages or broaden the range.")
        else:
            ctx_range = format_hits_for_context(_to_hits(range_items), limit=14)
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
                f"focus::{query}::{filters['date_from']}::{filters['date_to']}::{DEFAULT_MODEL}",
                sys_focus, usr_focus, model=DEFAULT_MODEL, max_tokens=PER_ITEM_TOKENS,
                temperature=0.2, stream=True)
            st.caption(f"Model: {used_model1}")

            sys2 = ("You analyze evolution across the full date range. Use ONLY context. "
                    "Be concrete: what gained emphasis, what was deemphasized, any NEW issues. "
                    "Cite with (Month YYYY — Title) based on headers.")
            usr2 = f"""Topic: {query}
Range: {filters['date_from'].isoformat()} → {filters['date_to'].isoformat()}

Context:
{ctx_range}

Task: Write a short 'Messaging Evolution' narrative for the range (6–10 sentences, crisp)."""
            st.markdown("### Messaging Evolution")
            (evol_md, used_model2) = llm_cached(
                f"evol::{query}::{filters['date_from']}::{filters['date_to']}::{DEFAULT_MODEL}",
                sys2, usr2, model=DEFAULT_MODEL, max_tokens=PER_ITEM_TOKENS,
                temperature=0.2, stream=True)
            st.caption(f"Model: {used_model2}")

            with st.expander("Show sources (Thematic Evolution)"):
                for (d, t, l) in sources_from_speeches(range_items):
                    st.markdown(f"- {d} — [{t}]({l})")

# ---------------- Optional Trajectory tab ----------------
if _HAS_P3:
    with tab_traj:
        if not speeches:
            st.info("Enter a search above to view trajectory analysis.")
        else:
            st.subheader("Messaging Trajectory")
            ctx = format_hits_for_context(_to_hits(page_items), limit=14)
            series = build_issue_trajectory(query, ctx, DEFAULT_MODEL)  # DataFrame: year, issue, share
            if not series.empty:
                fig = px.area(series, x="year", y="share", color="issue", groupnorm="fraction")
                st.plotly_chart(fig, use_container_width=True)
                forecast = forecast_issue_trends(series)
                if not forecast.empty:
                    st.markdown("**Next-year trend (simple forecast)**")
                    fig2 = px.bar(forecast, x="issue", y="forecast_share")
                    st.plotly_chart(fig2, use_container_width=True)
                nar = trajectory_narrative(query, ctx, DEFAULT_MODEL)
                st.markdown(nar)
            else:
                st.info("No trajectory could be derived from current context.")

# ---------------- Rapid Response tab ----------------
with tab_rr:
    st.subheader("Rapid Response")
    inquiry = st.text_area("Media inquiry from journalist", "")
    if st.button("Generate response") and inquiry.strip():
        try:
            rr_speeches, _ = retrieve_speeches(
                query=inquiry,
                index=index, metas=metas, chunks=chunks,
                filters=filters,
                sort="relevance",
                precision=high_precision,
                strictness=strictness,
                exact_phrase=True,
                exclude_terms=[],
                core_topic_only=core_topic_only,
                use_llm_rerank=True,
                model=DEFAULT_MODEL
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            rr_speeches = []

        rr_hits = _to_hits(rr_speeches[:30])
        ctx_rr = format_hits_for_context(rr_hits, limit=18)
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
            key = f"rr::{hashlib.sha256((inquiry+ctx_rr).encode()).hexdigest()}::{DEFAULT_MODEL}"
            (rr_md, used_model) = llm_cached(key, sys, usr, model=DEFAULT_MODEL,
                                             max_tokens=500, temperature=0.2, stream=True)
            st.caption(f"Model: {used_model}")
            with st.expander("Show sources (Rapid Response)"):
                for (d,t,l) in sources_from_speeches(rr_speeches):
                    st.markdown(f"- {d} — [{t}]({l})")
        else:
            st.warning("No relevant context found to answer the inquiry.")
    else:
        st.info("Paste the inquiry and click Generate response.")

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
        try:
            draft_speeches, _ = retrieve_speeches(
                query=big_ideas,
                index=index, metas=metas, chunks=chunks,
                filters=filters,
                sort=sort_key,
                precision=high_precision,
                strictness=strictness,
                exact_phrase=False,
                exclude_terms=[],
                core_topic_only=core_topic_only,
                use_llm_rerank=True,
                model=DEFAULT_MODEL,
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            draft_speeches = []

        draft_hits = _to_hits(draft_speeches[:30])
        ctx = format_hits_for_context(draft_hits, limit=18)

        params = []
        if audience.strip(): params.append(f"Audience: {audience}")
        if venue.strip(): params.append(f"Venue: {venue}")
        if tone.strip(): params.append(f"Tone: {tone}")
        if style.strip(): params.append(f"Style: {style}")
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
        (md, used_model) = llm_cached(key, sys, usr, model=DEFAULT_MODEL,
                                      max_tokens=1200, temperature=0.3, stream=True)
        st.caption(f"Model: {used_model}")
        st.download_button("Download draft (Markdown)", md.encode("utf-8"), file_name="speech_draft.md", mime="text/markdown")
    else:
        st.info("Provide big ideas and optional parameters, then click Draft outline.")
