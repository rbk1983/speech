# app_llm.py — main app that can build the index if missing
import os, textwrap
import streamlit as st

st.set_page_config(page_title="Kristalina Speech Intelligence (LLM)", layout="wide")
st.title("Kristalina Georgieva — LLM Speech Intelligence")

# --- helper: ensure we have an index; build it here if missing
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
        with st.spinner("Building index… this can take a few minutes on first run. Please keep this page open."):
            try:
                from build_index import main as build_index_main
                build_index_main()
                st.success("Index built successfully. Click **Rerun** (top-right) or refresh the page.")
                st.stop()
            except Exception as e:
                st.exception(e)
                st.stop()
    st.stop()

_ensure_index()  # Ensure index exists before we import retrieval utils

# Now that files exist, import the RAG helpers
from rag_utils import load_df, load_index, retrieve, llm

@st.cache_resource
def _load_everything():
    df = load_df()
    idx, metas, chunks = load_index()
    return df, idx, metas, chunks

df, index, metas, chunks = _load_everything()

# --- Controls
query = st.text_input("Search topic (e.g., “artificial intelligence”, “climate finance”)", "")
col1, col2, col3 = st.columns(3)
years = sorted(df["date"].dt.year.unique().tolist())
if not years:
    st.error("No dates found in your dataset.")
    st.stop()
year_a = col1.selectbox("Year A", years, index=0 if years else None)
year_b = col2.selectbox("Year B", years, index=min(1, len(years)-1) if len(years)>1 else 0)

all_themes = sorted({t for lst in df.get("new_themes", df.get("themes", [])) for t in (lst if isinstance(lst, list) else [])}) if ("new_themes" in df.columns or "themes" in df.columns) else []
theme_filter = st.multiselect("Theme (optional)", all_themes, default=[])

filters = {"themes": theme_filter} if theme_filter else {}

# --- Retrieve top chunks
if query.strip():
    hits = retrieve(query, index, metas, chunks, k=24, filters=filters)
    st.write(f"Found {len(hits)} relevant chunks.")
else:
    hits = []
    st.info("Enter a topic to begin.")

def join_chunks(hits_subset, limit=12, char_limit=900):
    out = []
    used = set()
    for (idx, m, ch) in hits_subset[:limit]:
        key = (m["date"], m["title"])
        if key in used:
            continue
        used.add(key)
        header = f"[{m['date']} — {m['title']}]({m['link']})"
        excerpt = textwrap.shorten(ch.replace("\n"," "), width=char_limit, placeholder="…")
        out.append(header + "\n" + excerpt)
    return "\n\n".join(out)

# --- Narrative Quick Compare (LLM)
st.header("Narrative Quick Compare (LLM)")
if hits:
    subset_a = [(i,m,c) for (i,m,c) in hits if int(m["year"]) == int(year_a)]
    subset_b = [(i,m,c) for (i,m,c) in hits if int(m["year"]) == int(year_b)]

    if not subset_a and not subset_b:
        st.warning("No retrieved content for either selected year with this query.")
    else:
        chunks_a = join_chunks(subset_a, limit=12)
        chunks_b = join_chunks(subset_b, limit=12)
        system = ("You are a senior IMF communications strategist. Use ONLY the provided context. "
                  "Focus on substance, not keywords. Avoid speculation. Include Month YYYY — Title where relevant.")
        user = f"""
Topic: {query}

Year {year_a} context:
{chunks_a}

Year {year_b} context:
{chunks_b}

Tasks:
1) For {year_a}: list 4–6 issue headings with 1–2 sentence summaries each (no quotes).
2) For {year_b}: list 4–6 issue headings with 1–2 sentence summaries each (no quotes).
3) Messaging evolution: a short, concrete narrative of what gained emphasis, what was deemphasized, and any new issues.
Return Markdown with three sections.
"""
        cmp_md = llm(system, user, model="gpt-4o-mini", max_tokens=900)
        st.markdown(cmp_md)
else:
    st.info("Quick Compare will appear after you search.")

# --- Top Quotes (LLM)
if hits:
    st.header("Top Quotes (LLM)")
    system = ("You are extracting quotes for media use. Use only provided context; give exact sentences. "
              "Each bullet must end with (Month YYYY — Title) and include the link.")
    user = f"""
Topic: {query}

Context:
{join_chunks(hits, limit=16)}

Task:
Return the 5 strongest quotes (1–3 sentences each) that directly address the topic.
"""
    quotes_md = llm(system, user, model="gpt-4o-mini", max_tokens=700)
    st.markdown(quotes_md)

# --- Briefing Pack (LLM)
if hits:
    st.header("Briefing Pack (LLM)")
    system = ("You are drafting a comms briefing. Use only the provided context and keep it concise and media-ready.")
    user = f"""
Topic: {query}

Context:
{join_chunks(hits, limit=18)}

Tasks:
- Executive summary (3–5 bullets).
- 5 key issues with 1–2 sentence summaries each.
- 5 strongest quotes (with Month YYYY — Title).
- Short timeline bullets (Month YYYY — Title) of notable remarks.
Return clean Markdown.
"""
    brief_md = llm(system, user, model="gpt-4o-mini", max_tokens=1200)
    st.markdown(brief_md)
