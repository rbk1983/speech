# app_llm.py — Phase 2.5: Ingestion + Rapid Response (URL & Reporter Question)
# Adds: Ingest tab for URLs, documents, audio/video; Rapid Response from URL / Reporter Question.
import os, io, datetime as _dt, hashlib
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
from ingest_utils import (
    extract_from_url, extract_from_pdf, extract_from_docx, extract_from_txt,
    transcribe_media, normalize_record, add_records_and_rebuild
)

st.set_page_config(page_title="Kristalina Speech Intelligence (LLM)", layout="wide")
st.title("Kristalina Georgieva — Speech Intelligence")

# ---------------- Ensure index exists ----------------
def _index_files_present():
    return all(os.path.exists(p) for p in ["index.faiss", "meta.json", "chunks.json"])

def _ensure_index():
    if _index_files_present():
        return True
    st.warning("Search index not found here. Build it once in this app (or use Ingest tab).")
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

# ---------------- Load data/index ----------------
@st.cache_resource
def _load_everything():
    df = load_df()
    idx, metas, chunks = load_index()
    return df, idx, metas, chunks

df, index, metas, chunks = _load_everything()

# ---------------- Sidebar controls ----------------
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
    date_to   = st.date_input("To", max_value=max_d, value=max_d)

    sort = st.radio("Sort by", ["Relevance", "Newest"], horizontal=True)

    view_mode = st.radio(
        "LLM Mode",
        ["Speed", "Depth"],
        help="Speed: gpt-4o-mini (smaller context). Depth: gpt-4o (more context).",
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
            date_to   = _dt.date.fromisoformat(s["to"])
            sort = s["sort"]
            view_mode = s["mode"]

# Optional theme facet
theme_col = "new_themes" if "new_themes" in df.columns else ("themes" if "themes" in df.columns else None)
all_themes = sorted({t for lst in (df[theme_col] if theme_col else []) for t in (lst if isinstance(lst, list) else [])}) if theme_col else []
theme_filter = st.multiselect("Filter by theme (optional)", all_themes, help="Click to filter results to these themes.")

filters = {"themes": theme_filter, "date_from": date_from, "date_to": date_to}

# ---------------- Model choice / context limits ----------------
if view_mode == "Depth":
    model_preferred = "gpt-4o"
    ctx_limit = 12
    per_item_tokens = 950
else:
    model_preferred = "gpt-4o-mini"
    ctx_limit = 8
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
    st.info("Enter a topic in the sidebar to begin.")
    st.stop()

sort_key = "newest" if sort == "Newest" else "relevance"
limit = page_size
offset = st.session_state.page_offset

hits, total = retrieve(query, index, metas, chunks, k=100, filters=filters, sort=sort_key, offset=offset, limit=limit)
st.caption(f"Showing {len(hits)} of {total} results — page {offset//limit + 1}")

# Cap how many items feed the LLM sections (for speed)
hits_for_llm = hits[: (28 if view_mode == "Depth" else 16)]

# Pagination buttons
c1, c2, _ = st.columns([1,1,6])
with c1:
    if st.button("◀ Prev", disabled=offset==0):
        st.session_state.page_offset = max(0, offset - limit); st.rerun()
with c2:
    if st.button("Next ▶", disabled=offset + limit >= total):
        st.session_state.page_offset = offset + limit; st.rerun()

# ---------------- Tabs ----------------
tab_res, tab_compare, tab_quotes, tab_brief, tab_viz, tab_align, tab_rr, tab_tone, tab_draft, tab_stake, tab_ingest = st.tabs(
    ["Results", "Thematic Evolution", "Top Quotes", "Briefing Pack", "Analytics",
     "Alignment", "Rapid Response", "Tone", "Draft Assist", "Stakeholders", "Ingest (+Rebuild)"]
)

# ---------------- Results tab ----------------
with tab_res:
    st.subheader("Most Relevant Speeches" if sort_key=="relevance" else "Most Recent Speeches")
    for (_, m, ch) in hits:
        st.markdown(f"**{m.get('date')} — [{m.get('title')}]({m.get('link')})**")
        sys = "You are an IMF speech analyst. Produce a concise snippet and 3–5 bullets grounded only in the excerpt."
        usr = f"Excerpt:\n{ch}\n\nReturn a 1–2 sentence snippet, then 3–5 bullets of key points."
        key = f"res::{m.get('date')}::{m.get('title')}::{model_preferred}"
        (md, used_model) = llm_cached(key, sys, usr, model=model_preferred, max_tokens=per_item_tokens, temperature=0.2)
        st.markdown(md)
        st.caption(f"Model: {used_model}")
    with st.expander("Show sources used on this page"):
        for (d,t,l) in sources_from_hits(hits):
            st.markdown(f"- {d} — [{t}]({l})")

# ---------------- Thematic Evolution (single date range) ----------------
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
        by_year = {}
        for (idx, m, ch) in range_hits:
            y = int(m.get("year", 0) or str(m.get("date",""))[:4] or 0)
            if y == 0: continue
            by_year.setdefault(y, []).append((idx, m, ch))
        years_asc = sorted(by_year.keys())

        per_year_limit = max(3, min(6, (12 if view_mode=="Depth" else 8) // max(1, len(years_asc))))
        year_sections = []
        for y in years_asc:
            ctx = format_hits_for_context(by_year[y], limit=per_year_limit, char_limit=900)
            if ctx.strip():
                year_sections.append(f"=== Year {y} ===\n{ctx}")
        full_ctx = "\n\n".join(year_sections) if year_sections else "(no matching context)"

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
        (per_year_md, used_model1) = llm_cached(f"peryear::{query}::{date_from}::{date_to}::{model_preferred}", sys1, usr1, model=model_preferred, max_tokens=per_item_tokens, temperature=0.2)
        st.markdown("### Per-Year Focus")
        st.markdown(per_year_md)
        st.caption(f"Model: {used_model1}")

        sys2 = ("You analyze evolution across the full date range. Use ONLY context. "
                "Be concrete: what gained emphasis, what was deemphasized, any NEW issues. "
                "Cite with (Month YYYY — Title) based on headers.")
        usr2 = f"""
Topic: {query}
Range: {date_from.isoformat()} → {date_to.isoformat()}

Use the same context as above.

Task: Write a short 'Messaging Evolution' narrative for the whole range (6–10 sentences, crisp).
"""
        (evol_md, used_model2) = llm_cached(f"evol::{query}::{date_from}::{date_to}::{model_preferred}", sys2, usr2, model=model_preferred, max_tokens=per_item_tokens, temperature=0.2)
        st.markdown("### Messaging Evolution")
        st.markdown(evol_md)
        st.caption(f"Model: {used_model2}")

        with st.expander("Show sources (Thematic Evolution)"):
            for (d, t, l) in sources_from_hits(range_hits):
                st.markdown(f"- {d} — [{t}]({l})")

# ---------------- Top Quotes tab ----------------
with tab_quotes:
    st.subheader("Top Quotes (LLM)")
    ctx_quotes = format_hits_for_context(hits_for_llm, limit=(ctx_limit+2))
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
    (quotes_md, used_model3) = llm_cached(f"quotes::{query}::{date_from}::{date_to}::{model_preferred}", sys_q, usr_q, model=model_preferred, max_tokens=per_item_tokens, temperature=0.2)
    st.markdown(quotes_md)
    st.caption(f"Model: {used_model3}")
    with st.expander("Show sources (Quotes)"):
        for (d,t,l) in sources_from_hits(hits_for_llm):
            st.markdown(f"- {d} — [{t}]({l})")

# ---------------- Briefing Pack tab ----------------
with tab_brief:
    st.subheader("Briefing Pack (LLM)")
    ctx_brief = format_hits_for_context(hits_for_llm, limit=(ctx_limit+4))
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
    (brief_md, used_model4) = llm_cached(f"brief::{query}::{date_from}::{date_to}::{model_preferred}", sys_b, usr_b, model=model_preferred, max_tokens=per_item_tokens+200, temperature=0.25)
    st.markdown(brief_md)
    st.caption(f"Model: {used_model4}")
    st.download_button("Download briefing (Markdown)", brief_md.encode("utf-8"), file_name="briefing.md", mime="text/markdown")

# ---------------- Analytics tab (visuals) ----------------
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

# ---------------- Alignment tab ----------------
with tab_align:
    st.subheader("Message Consistency & Alignment")
    playbook = load_playbook()
    ctx = format_hits_for_context(hits_for_llm, limit=(ctx_limit+2))

    # Radar/bar + narrative
    radar_df = alignment_radar_data(query, playbook, ctx, model_preferred)
    if radar_df is not None and not radar_df.empty and set(["issue","score"]).issubset(radar_df.columns):
        try:
            if len(radar_df) >= 3:
                fig = px.line_polar(radar_df, r="score", theta="issue", line_close=True)
                fig.update_traces(fill='toself')
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(radar_df, x="issue", y="score")
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.dataframe(radar_df)
    else:
        st.info("No issues detected for radar. Try widening the date range or query.")

    nar = alignment_narrative(query, playbook, ctx, model_preferred)
    st.markdown(nar)

# ---------------- Rapid Response tab ----------------
with tab_rr:
    st.subheader("Rapid Response")

    # A) Generate from live news URL
    st.markdown("#### From a News URL")
    url_in = st.text_input("Paste a news article URL")
    if st.button("Generate press lines from URL"):
        if not url_in.strip():
            st.warning("Please paste a URL first.")
        else:
            art = extract_from_url(url_in.strip())
            if not art or not art.get("text"):
                st.error("Could not extract content from this URL. Try another link or paste the text manually.")
            else:
                st.success(f"Pulled: {art.get('title','(untitled)')} — {art.get('date','unknown date')}")
                ctx = format_hits_for_context(hits_for_llm, limit=(ctx_limit+4))
                sys = ("You write rapid-response lines grounded ONLY in the provided IMF speech context AND the article content. "
                       "No speculation; if unknown, say 'we are assessing'. Tone: calm, factual, forward-looking.")
                usr = f"""Article title: {art.get('title')}
Article date: {art.get('date')}
Article source text (trimmed):
{art.get('text')[:4000]}

IMF speech context:
{ctx}

Tasks:
- 4–6 press lines (bulleted), 1 sentence each.
- 3 policy specifics with (Month YYYY — Title) from IMF context.
- 3 reporter Q&As (Q + short A grounded in context; if not addressed previously, say so and bridge to what we have said).

Return clean Markdown.
"""
                (pack_md, used_model) = llm_cached(f"rr_url::{url_in}::{query}", sys, usr, model=model_preferred, max_tokens=900, temperature=0.2)
                st.markdown(pack_md)
                st.caption(f"Model: {used_model}")
                st.download_button("Download press lines (Markdown)", pack_md.encode("utf-8"), file_name="press_lines.md", mime="text/markdown")

    st.divider()

    # B) Generate from reporter question
    st.markdown("#### From a Reporter Question")
    rq = st.text_area("Paste the reporter's question")
    if st.button("Generate press lines from question"):
        if not rq.strip():
            st.warning("Please paste the reporter question first.")
        else:
            ctx = format_hits_for_context(hits_for_llm, limit=(ctx_limit+6))
            sys = ("You craft a comprehensive press packet in the speaker's voice grounded ONLY in past remarks. "
                   "If we haven't addressed the issue directly, say so and propose a bridging line consistent with our established positions.")
            usr = f"""Reporter question:
{rq}

Grounding excerpts:
{ctx}

Tasks:
- 4–6 press lines (1 sentence each), grounded in context. If not directly addressed historically, say so and provide a careful bridge.
- 3–4 key facts/policy specifics with (Month YYYY — Title).
- 3 Q&As (Q + short A, with references).
Return clean Markdown.
"""
            (pack_md, used_model) = llm_cached(f"rr_q::{hashlib.sha256(rq.encode()).hexdigest()}::{query}", sys, usr, model=model_preferred, max_tokens=950, temperature=0.25)
            st.markdown(pack_md)
            st.caption(f"Model: {used_model}")
            st.download_button("Download press packet (Markdown)", pack_md.encode("utf-8"), file_name="press_packet.md", mime="text/markdown")

# ---------------- Tone tab ----------------
with tab_tone:
    st.subheader("Sentiment & Tone")
    ctx = format_hits_for_context(hits_for_llm, limit=(ctx_limit+6))
    ts = tone_time_series(query, ctx, model_preferred)
    if not ts.empty:
        fig = px.line(ts, x="date", y="score", color="tone", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    hm = tone_heatmap_data(query, ctx, model_preferred)
    if not hm.empty:
        fig2 = px.imshow(hm.pivot_table(index="tone", columns="year", values="score", fill_value=0),
                         aspect="auto", color_continuous_scale="Blues")
        st.plotly_chart(fig2, use_container_width=True)

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
        ctx = format_hits_for_context(hits_for_llm, limit=(ctx_limit+6))
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
    stakeholders = load_stakeholders()
    ctx = format_hits_for_context(hits_for_llm, limit=(ctx_limit+4))
    scores = stakeholder_scores(query, stakeholders, ctx, model_preferred)  # returns DataFrame
    if not scores.empty:
        fig = px.bar(scores, x="stakeholder", y="score")
        st.plotly_chart(fig, use_container_width=True)
        nar = stakeholder_narrative(query, stakeholders, ctx, model_preferred)
        st.markdown(nar)
    else:
        st.info("No clear stakeholder mapping detected—try a broader query or range.")

# ---------------- Ingest tab ----------------
with tab_ingest:
    st.subheader("Add New Content to Corpus & Rebuild Index")

    st.markdown("**Option A — Add by URL**")
    colu1, colu2 = st.columns([3,1])
    new_url = colu1.text_input("Speech/interview URL", placeholder="https://...")
    fetch_btn = colu2.button("Fetch & Preview")
    if fetch_btn and new_url.strip():
        art = extract_from_url(new_url.strip())
        if not art or not art.get("text"):
            st.error("Could not extract from this URL. If it's behind a paywall or script-heavy, paste the text instead.")
        else:
            st.success("Fetched content. Review and edit as needed before saving.")
            with st.form("save_from_url"):
                title = st.text_input("Title", art.get("title","(untitled)"))
                date  = st.date_input("Date", value=_dt.date.fromisoformat(art.get("date")) if art.get("date") else _dt.date.today())
                link  = st.text_input("Link", new_url.strip())
                location = st.text_input("Location (optional)", "")
                raw = st.text_area("Text (you can edit)", art.get("text",""), height=200)
                themes = st.text_input("Themes (comma-separated, optional)", "")
                submitted = st.form_submit_button("Add to corpus")
                if submitted:
                    rec = normalize_record(title=title, date=str(date), link=link, location=location, transcript=raw, themes=themes)
                    ok, msg = add_records_and_rebuild([rec])
                    if ok:
                        st.success(msg + " — Re-run your search to see new content.")
                    else:
                        st.error(msg)

    st.divider()
    st.markdown("**Option B — Upload a File**")
    up = st.file_uploader("PDF, DOCX, TXT, MP3, M4A, WAV, MP4, WEBM", type=["pdf","docx","txt","mp3","m4a","wav","mp4","webm"])
    if up is not None:
        ext = (up.name.split(".")[-1] or "").lower()
        text = ""
        try:
            if ext == "pdf":
                text = extract_from_pdf(up.read())
            elif ext == "docx":
                text = extract_from_docx(up.read())
            elif ext == "txt":
                text = extract_from_txt(up.read())
            elif ext in ("mp3","m4a","wav","mp4","webm"):
                text = transcribe_media(up.read(), up.name)
            else:
                st.error("Unsupported file type.")
        except Exception as e:
            st.exception(e)

        if text:
            st.success(f"Extracted ~{len(text.split())} words.")
            with st.form("save_from_upload"):
                title = st.text_input("Title", up.name.rsplit(".",1)[0])
                date  = st.date_input("Date", value=_dt.date.today())
                link  = st.text_input("Link (optional)", "")
                location = st.text_input("Location (optional)", "")
                raw = st.text_area("Text (you can edit)", text, height=200)
                themes = st.text_input("Themes (comma-separated, optional)", "")
                submitted2 = st.form_submit_button("Add to corpus")
                if submitted2:
                    rec = normalize_record(title=title, date=str(date), link=link, location=location, transcript=raw, themes=themes)
                    ok, msg = add_records_and_rebuild([rec])
                    if ok:
                        st.success(msg + " — Re-run your search to see new content.")
                    else:
                        st.error(msg)

    st.info("Note: Rebuilding the index embeds and reindexes all content (fast, but may take ~30–60s). Ensure OPENAI_API_KEY is set.")
