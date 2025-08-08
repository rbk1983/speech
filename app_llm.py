# app_llm.py — LLM app with thematic Quick Compare (issues + evolution)
import os, textwrap, datetime as _dt
import streamlit as st
from rag_utils import load_df, load_index, retrieve, llm

st.set_page_config(page_title="Kristalina Speech Intelligence (LLM)", layout="wide")
st.title("Kristalina Georgieva — LLM Speech Intelligence")

# --------- Ensure index exists (build-in-app if missing) ----------
def _index_files_present():
    return all(os.path.exists(p) for p in ["index.faiss", "meta.json", "chunks.json"])

def _ensure_index():
    if _index_files_present():
        return True
    st.warning("Search index not found in this app instance. Click below to build it here (one-time).")
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. In Streamlit Cloud, go to Settings → Secrets and add it, then redeploy.")
        st.stop()
    if st.button("Build index now"):
        with st.spinner("Building index… this can take a few minutes on first run. Keep this page open."):
            try:
                from build_index import main as build_index_main
                build_index_main()
                st.success("Index built successfully. Click **Rerun** (top-right) or refresh the page.")
                st.stop()
            except Exception as e:
                st.exception(e)
                st.stop()
    st.stop()

_ensure_index()

# --------- Load data/index ----------
@st.cache_resource
def _load_everything():
    df = load_df()
    idx, metas, chunks = load_index()
    return df, idx, metas, chunks

df, index, metas, chunks = _load_everything()

# --------- Controls ----------
query = st.text_input("Search topic (e.g., “artificial intelligence”, “climate finance”)", "")
col1, col2 = st.columns(2)
years = sorted(df["date"].dt.year.unique().tolist())
if not years:
    st.error("No dates found in your dataset.")
    st.stop()
default_b = min(1, len(years)-1) if len(years) > 1 else 0
year_a = col1.selectbox("Year A", years, index=0)
year_b = col2.selectbox("Year B", years, index=default_b)

# Optional theme filter
if ("new_themes" in df.columns) or ("themes" in df.columns):
    all_themes = sorted({t for lst in df.get("new_themes", df.get("themes", [])) for t in (lst if isinstance(lst, list) else [])})
else:
    all_themes = []
theme_filter = st.multiselect("Theme (optional)", all_themes, default=[])
filters = {"themes": theme_filter} if theme_filter else {}

# --------- Retrieval ----------
def _date_key(meta_date):
    try:
        return _dt.datetime.fromisoformat(str(meta_date))
    except Exception:
        return _dt.datetime.min

if query.strip():
    hits = retrieve(query, index, metas, chunks, k=24, filters=filters)
    hits = sorted(hits, key=lambda t: _date_key(t[1].get("date","")), reverse=True)
    st.caption(f"Retrieved {len(hits)} results (deduped to one best chunk per speech).")
else:
    st.info("Enter a topic to begin.")
    st.stop()

# Group for Quick Compare
year_a_hits = [(i,m,c) for (i,m,c) in hits if int(m.get("year", 0)) == int(year_a)]
year_b_hits = [(i,m,c) for (i,m,c) in hits if int(m.get("year", 0)) == int(year_b)]

# Helper: format chunks with headers so LLM can cite
def _format_hits_for_context(hit_list, limit=12, char_limit=900):
    """
    Each item becomes:
    [YYYY-MM-DD — Title](link)
    excerpt...
    """
    out = []
    used = set()
    for (idx, m, ch) in hit_list[:limit]:
        key = (m.get("date"), m.get("title"))
        if key in used:
            continue
        used.add(key)
        header = f"[{m.get('date')} — {m.get('title')}]({m.get('link')})"
        excerpt = textwrap.shorten((ch or "").replace("\n", " "), width=char_limit, placeholder="…")
        out.append(header + "\n" + excerpt)
    return "\n\n".join(out) if out else "(no matching context)"

# --------- Thematic Quick Compare (LLM) ----------
st.header("Quick Compare — Thematic, LLM-written")
if not year_a_hits and not year_b_hits:
    st.warning("No retrieved content for either selected year with this query. Try removing the Theme filter or broadening your query.")
else:
    ctx_a = _format_hits_for_context(year_a_hits, limit=12)
    ctx_b = _format_hits_for_context(year_b_hits, limit=12)

    system = (
        "You are a senior IMF communications strategist. "
        "Using ONLY the provided context, write issue-focused analysis that is precise, sober, and media-ready. "
        "Focus on substance (policies, rationales, risks, instruments), not keywords. Avoid speculation. "
        "When you reference a point, add (Month YYYY — Title) right after it using the bracket headers."
    )
    user = f"""
Topic: {query}

Year {year_a} context:
{ctx_a}

Year {year_b} context:
{ctx_b}

Tasks:
1) For {year_a}: list 4–6 **issue** headings with 1–2 sentence summaries each (no quotes). Keep it concise and substantive.
2) For {year_b}: list 4–6 **issue** headings with 1–2 sentence summaries each (no quotes).
3) **Messaging evolution**: a short, concrete narrative describing what gained emphasis, what was deemphasized, and any **new issues** that emerged. Reference specific speeches using (Month YYYY — Title).
Return clean Markdown with three sections: "{year_a} Focus", "{year_b} Focus", "Messaging Evolution".
"""
    cmp_md = llm(system, user, model="gpt-4o-mini", max_tokens=900, temperature=0.3)
    st.markdown(cmp_md)

# --------- Top Quotes (LLM) ----------
st.header("Top Quotes (LLM)")
system_q = (
    "You are extracting quotes for media use. Use ONLY the provided context; return exact sentences. "
    "Each bullet must end with (Month YYYY — Title) and include the link from the header."
)
user_q = f"""
Topic: {query}

Context:
{_format_hits_for_context(hits, limit=16)}

Task:
Return the 5 strongest, most on-topic quotes (1–3 sentences each). Each bullet ends with (Month YYYY — Title).
"""
quotes_md = llm(system_q, user_q, model="gpt-4o-mini", max_tokens=700, temperature=0.3)
st.markdown(quotes_md)

# --------- Briefing Pack (LLM) ----------
st.header("Briefing Pack (LLM)")
system_b = "You are drafting a comms briefing. Use only the provided context and keep it concise and media-ready."
user_b = f"""
Topic: {query}

Context:
{_format_hits_for_context(hits, limit=18)}

Tasks:
- Executive summary (3–5 bullets).
- 5 key issues with 1–2 sentence summaries each (no quotes).
- 5 strongest quotes (with Month YYYY — Title).
- Short timeline bullets (Month YYYY — Title) of notable remarks.
Return clean Markdown (no extras).
"""
brief_md = llm(system_b, user_b, model="gpt-4o-mini", max_tokens=1200, temperature=0.3)
st.markdown(brief_md)
