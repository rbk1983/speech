# app_llm.py
# Kristalina Georgieva Speeches — RAG Console (PKL-first corpus)
# - Auto-ingest from ./speeches_data.pkl on first load (if present)
# - Strict corpus-only toggle (air-gapped answers)
# - Tabs: Results, Thematic Evolution, Rapid Response, Draft Assist, Corpus
# - Manual upload/ingest to extend corpus; delete by source_id
# - Re-import from PKL on demand; dedupe by digest
#
# Setup:
#   pip install streamlit openai pypdf python-docx pandas
#   export OPENAI_API_KEY=your_key_here
# Run:
#   streamlit run app_llm.py

import os
import io
import re
import json
import pickle
import hashlib
import unicodedata
from datetime import datetime, date
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import streamlit as st

# -------- OpenAI client --------
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

# -------- Optional parsers --------
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

try:
    import docx  # python-docx
    HAS_DOXC = True
except Exception:
    HAS_DOXC = False

# -------- Config --------
APP_TITLE = "Kristalina Georgieva Speeches — RAG"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o")
EMBED_DIM = 1536  # for text-embedding-3-small

DATA_DIR = os.environ.get("KG_DATA_DIR", "kg_speeches_store")
INDEX_VEC = os.path.join(DATA_DIR, "embeddings.npy")
INDEX_META = os.path.join(DATA_DIR, "metadata.json")
PKL_PATH = os.environ.get("KG_PKL_PATH", "speeches_data.pkl")  # main corpus file (if present)

# -------- Utilities --------
def ensure_store():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(INDEX_VEC):
        np.save(INDEX_VEC, np.zeros((0, EMBED_DIM), dtype=np.float32))
    if not os.path.exists(INDEX_META):
        with open(INDEX_META, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)

def load_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    ensure_store()
    vecs = np.load(INDEX_VEC)
    with open(INDEX_META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return vecs, meta

def save_index(vecs: np.ndarray, meta: List[Dict[str, Any]]):
    ensure_store()
    np.save(INDEX_VEC, vecs.astype(np.float32))
    with open(INDEX_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace("\r", "\n").strip()

def chunk_text(text: str, max_tokens: int = 700, overlap_tokens: int = 100) -> List[str]:
    # Approximate tokens with characters (~4 chars/token)
    approx = max_tokens * 4
    overlap = overlap_tokens * 4
    t = normalize_text(text)
    chunks, i, n = [], 0, len(t)
    while i < n:
        j = min(i + approx, n)
        chunk = t[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def digest_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def file_digest(name: str, data: bytes) -> str:
    h = hashlib.sha256()
    h.update(name.encode("utf-8"))
    h.update(data)
    return h.hexdigest()[:16]

def try_parse_date_from_text(text: str) -> Optional[str]:
    """
    Try to find a date like 'January 12, 2024' or '2024-01-12' or '12 January 2024' in the text.
    Returns ISO date 'YYYY-MM-DD' if found.
    """
    # ISO
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", text)
    if m:
        y, mo, d = m.groups()
        try:
            return str(date(int(y), int(mo), int(d)))
        except Exception:
            pass

    # Month D, YYYY  or D Month YYYY
    months = "(January|February|March|April|May|June|July|August|September|October|November|December)"
    m = re.search(rf"{months}\s+(\d{{1,2}}),\s*(20\d{{2}})", text)
    if m:
        mon, d, y = m.groups()
        try:
            dt = datetime.strptime(f"{mon} {d} {y}", "%B %d %Y").date()
            return str(dt)
        except Exception:
            pass
    m = re.search(rf"(\d{{1,2}})\s+{months}\s+(20\d{{2}})", text)
    if m:
        d, mon, y = m.groups()
        try:
            dt = datetime.strptime(f"{d} {mon} {y}", "%d %B %Y").date()
            return str(dt)
        except Exception:
            pass
    return None

# -------- Content readers --------
def read_file_to_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith((".txt", ".md")):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode("latin-1", errors="ignore")
    if name.endswith(".pdf"):
        if not HAS_PYPDF:
            st.warning("`pypdf` not installed. Skipping PDF. Run: pip install pypdf")
            return ""
        text = []
        reader = PdfReader(io.BytesIO(data))
        for page in reader.pages:
            try:
                text.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(text)
    if name.endswith(".docx"):
        if not HAS_DOXC:
            st.warning("`python-docx` not installed. Skipping DOCX. Run: pip install python-docx")
            return ""
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    # Fallback
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# -------- Embeddings --------
def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    if not OPENAI_OK:
        raise RuntimeError("OpenAI package not available. Install with `pip install openai`.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set.")
    embs = []
    B = 64
    for i in range(0, len(texts), B):
        batch = texts[i:i + B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embs.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
    return np.vstack(embs) if embs else np.zeros((0, EMBED_DIM), dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.array([])
    a_norm = np.linalg.norm(a, axis=1) + 1e-9
    b_norm = np.linalg.norm(b) + 1e-9
    return (a @ b) / (a_norm * b_norm)

# -------- PKL ingestion --------
def _iter_pkl_records(obj: Any) -> List[Dict[str, Any]]:
    """
    Normalize various PKL shapes to a list of dicts with keys: text, title?, date_iso?, source_name?
    Accepts: list[dict], pandas.DataFrame, dict of lists, dict with 'records' key, etc.
    """
    records: List[Dict[str, Any]] = []
    try:
        import pandas as pd  # optional
        is_pd = True
    except Exception:
        is_pd = False

    # list of dicts
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj

    # pandas DataFrame
    if is_pd and "DataFrame" in str(type(obj)):
        for _, row in obj.iterrows():  # type: ignore
            d = {k: row[k] for k in row.keys()}
            records.append({k: (None if d[k] is None else str(d[k])) for k in d})
        return records

    # dict of lists
    if isinstance(obj, dict):
        # if it has 'records' key
        if "records" in obj and isinstance(obj["records"], list):
            return obj["records"]
        # align len and zip
        keys = list(obj.keys())
        if keys and isinstance(obj[keys[0]], list):
            length = len(obj[keys[0]])
            for i in range(length):
                rec = {k: obj[k][i] if i < len(obj[k]) else None for k in keys}
                records.append(rec)
            return records

    # fallback: wrap as single record
    records = [obj] if isinstance(obj, dict) else []
    return records

def _build_doc_id(title: Optional[str], date_iso: Optional[str], text: str) -> str:
    title_part = (title or "").strip()
    date_part = (date_iso or "").strip()
    key = f"{title_part}|{date_part}|{text.strip()[:5000]}"  # limit to first 5k chars for stability
    return digest_str(key)

def ingest_pkl(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {"loaded": False, "new_docs": 0, "new_chunks": 0}
    with open(path, "rb") as f:
        obj = pickle.load(f)
    records = _iter_pkl_records(obj)
    if not records:
        return {"loaded": False, "new_docs": 0, "new_chunks": 0}

    vecs, meta = load_index()
    existing_ids = {m.get("source_id") for m in meta}

    to_texts: List[str] = []
    to_meta: List[Dict[str, Any]] = []

    for rec in records:
        text = str(rec.get("text") or rec.get("content") or "").strip()
        if not text:
            continue
        # normalize metadata
        title = rec.get("title") or rec.get("headline") or rec.get("name") or ""
        # Date could be 'date', 'date_iso', 'published', etc.
        raw_date = (rec.get("date_iso") or rec.get("date") or rec.get("published") or "").strip()
        iso = None
        if raw_date:
            # try to normalize common formats
            try:
                # If already YYYY-MM-DD
                datetime.strptime(raw_date, "%Y-%m-%d")
                iso = raw_date
            except Exception:
                # try parsing via simple heuristics
                maybe = try_parse_date_from_text(raw_date)
                if not maybe:
                    maybe = try_parse_date_from_text(text[:400])
                iso = maybe
        else:
            iso = try_parse_date_from_text(text[:400])

        source_name = rec.get("source_name") or rec.get("source") or rec.get("file_name") or (title or "pkl_record")

        doc_id = _build_doc_id(str(title), iso, text)
        if doc_id in existing_ids:
            continue

        # chunk + stage
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            to_meta.append({
                "id": f"{doc_id}:{i}",
                "chunk_id": i,
                "source_id": doc_id,
                "source_name": str(source_name),
                "title": str(title) if title is not None else None,
                "date_iso": iso,
                "text": ch,
            })
            to_texts.append(ch)

    if not to_texts:
        return {"loaded": True, "new_docs": 0, "new_chunks": 0}

    client = OpenAI()
    new_vecs = embed_texts(client, to_texts)

    if vecs.size == 0:
        merged_vecs = new_vecs
        merged_meta = to_meta
    else:
        merged_vecs = np.vstack([vecs, new_vecs])
        merged_meta = meta + to_meta

    save_index(merged_vecs, merged_meta)
    return {"loaded": True, "new_docs": len({m['source_id'] for m in to_meta}), "new_chunks": len(to_texts)}

# -------- Retrieval --------
def retrieve(query: str, k: int, score_threshold: float = 0.12,
             start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[Dict[str, Any]]:
    vecs, meta = load_index()
    if vecs.size == 0 or not meta:
        return []
    client = OpenAI()
    q_vec = embed_texts(client, [query])[0]
    sims = cosine_sim(vecs, q_vec)
    order = np.argsort(-sims)  # high to low similarity

    # Build results with optional date filtering
    results: List[Dict[str, Any]] = []
    for i in order[: max(k * 5, k)]:  # oversample then threshold
        score = float(sims[i])
        if score < score_threshold:
            continue
        m = meta[i].copy()
        # date filter: parse item date if present
        iso = m.get("date_iso")
        if iso:
            try:
                d = datetime.strptime(iso, "%Y-%m-%d").date()
                if start_date and d < start_date:
                    continue
                if end_date and d > end_date:
                    continue
            except Exception:
                pass
        m["score"] = score
        results.append(m)

    # Sort newest first when dates are available
    def _key(x):
        iso = x.get("date_iso")
        try:
            return datetime.strptime(iso, "%Y-%m-%d")
        except Exception:
            return datetime.min
    results.sort(key=_key, reverse=True)
    return results[:k]

def _fmt_src(item: Dict[str, Any]) -> str:
    return (
        item.get("source_name")
        or item.get("source")
        or item.get("file_name")
        or item.get("doc_name")
        or "unknown"
    )

def _fmt_chunk_id(item: Dict[str, Any]) -> str:
    return str(item.get("chunk_id", item.get("id", "?")))

def _fmt_text(item: Dict[str, Any]) -> str:
    return (item.get("text") or item.get("content") or "").strip()

def _has_context(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())

def _format_context_blocks(chunks: List[Dict[str, Any]]) -> str:
    blocks = []
    for c in chunks:
        header = []
        if c.get("date_iso"):
            header.append(c["date_iso"])
        if c.get("title"):
            header.append(c["title"])
        head = " — ".join(header) if header else f"{_fmt_src(c)} | chunk {_fmt_chunk_id(c)}"
        blocks.append(f"[{head}]\n{_fmt_text(c)}")
    return "\n\n---\n\n".join(blocks)

# -------- Chat helpers --------
def chat_strict(user_query: str, chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "Not found in corpus."
    if not OPENAI_OK or not os.environ.get("OPENAI_API_KEY"):
        return "Error: OpenAI unavailable or API key not set."
    client = OpenAI()
    system_msg = (
        "You are a meticulous IMF speech analyst. "
        "Answer the user's question using ONLY the provided context blocks. "
        "If the answer is not fully supported by the context, reply exactly: Not found in corpus."
    )
    content = f"Context:\n\n{_format_context_blocks(chunks)}\n\n---\n\nQuestion: {user_query}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": content}],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()

def chat_blended(user_query: str, chunks: List[Dict[str, Any]]) -> str:
    if not OPENAI_OK or not os.environ.get("OPENAI_API_KEY"):
        return "Error: OpenAI unavailable or API key not set."
    client = OpenAI()
    system_msg = (
        "You are a helpful communications analyst. "
        "Use the provided context when relevant, but you may use your broader knowledge. "
        "When you rely on the context, cite the date and title in parentheses."
    )
    context = _format_context_blocks(chunks) if chunks else "(no retrieved context)"
    content = f"Context (may be partial):\n\n{context}\n\n---\n\nQuestion: {user_query}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": content}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()

# -------- Streamlit UI --------
st.set_page_config(page_title=APP_TITLE, page_icon="🗂️", layout="wide")
st.title("🗂️ " + APP_TITLE)

# On first load: try ingesting from speeches_data.pkl
if "pkl_ingested_once" not in st.session_state:
    st.session_state["pkl_ingested_once"] = True
    if os.path.isfile(PKL_PATH):
        try:
            with st.spinner(f"Ingesting corpus from `{PKL_PATH}`…"):
                stats = ingest_pkl(PKL_PATH)
            if stats.get("loaded"):
                st.success(f"Corpus loaded. New docs: {stats['new_docs']}, new chunks: {stats['new_chunks']}.")
            else:
                st.info(f"No usable records found in `{PKL_PATH}`.")
        except Exception as e:
            st.error(f"PKL ingestion failed: {e}")
    else:
        st.info(f"No `{PKL_PATH}` found. You can upload content below and/or set KG_PKL_PATH.")

with st.sidebar:
    st.header("Settings")
    strict_mode = st.toggle(
        "Strict corpus-only mode",
        value=True,
        help=("ON: Answer ONLY from retrieved speech excerpts. "
              "If nothing is relevant, returns 'Not found in corpus.'\n"
              "OFF: Model may use its general knowledge in addition to context.")
    )
    top_k = st.slider("Top-K chunks per query", 1, 20, 8, 1)
    score_threshold = st.slider(
        "Min similarity (retrieval filter)",
        0.00, 0.50, 0.12, 0.01,
        help="Raise to be more conservative (fewer but surer matches)."
    )

    st.markdown("---")
    st.caption("Date range filter (optional, uses 'date_iso' from metadata):")
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start", value=None)
    with c2:
        end = st.date_input("End", value=None)

# ---- Tabs ----
tab_results, tab_evolution, tab_rr, tab_draft, tab_corpus = st.tabs(
    ["Results", "Thematic Evolution", "Rapid Response", "Draft Assist", "Corpus"]
)

# ===== Results Tab =====
with tab_results:
    st.subheader("🔎 Query Speeches")
    query = st.text_input("Ask a question (summaries, talking points, citations, etc.)", key="q_results")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        run = st.button("Run", type="primary", key="run_results")
    with col_b:
        show_ctx = st.checkbox("Show retrieved context", value=True, key="show_ctx_results")

    if run and query.strip():
        with st.spinner("Retrieving..."):
            results = retrieve(query=query, k=top_k, score_threshold=score_threshold,
                               start_date=start, end_date=end)

        # Viewer
        st.markdown(f"**Retrieved {len(results)} chunk(s)**")
        if show_ctx:
            with st.expander("View retrieved chunks"):
                for r in results:
                    src = _fmt_src(r)
                    cid = _fmt_chunk_id(r)
                    score = r.get("score", None)
                    score_txt = f" — score {score:.3f}" if isinstance(score, (int, float)) else ""
                    date_s = r.get("date_iso") or "Unknown date"
                    title = r.get("title") or src
                    preview = _fmt_text(r)[:1000]
                    more = "…" if len(_fmt_text(r)) > 1000 else ""
                    st.markdown(f"**{date_s} — {title}** ({src} — chunk {cid}{score_txt})\n\n{preview}{more}")
                    st.markdown("---")

        # Strict mode hard-stop if no results
        if run:
            if strict_mode and not results:
                st.markdown("### 🧠 Answer")
                st.code("Not found in corpus.")
            elif run:
                # Build chat messages and answer
                with st.spinner("Generating answer..."):
                    answer = chat_strict(query, results) if strict_mode else chat_blended(query, results)

                st.markdown("### 🧠 Answer")
                if strict_mode and answer.strip().lower() == "not found in corpus.":
                    st.code("Not found in corpus.")
                else:
                    st.write(answer)

# ===== Thematic Evolution Tab =====
with tab_evolution:
    st.subheader("📈 Thematic Evolution")
    theme_q = st.text_input("Theme or query (e.g., 'climate finance', 'debt restructuring')", key="q_evol")
    run_e = st.button("Analyze", key="run_evol")
    if run_e and theme_q.strip():
        with st.spinner("Retrieving theme-related context..."):
            hits = retrieve(query=theme_q, k=max(20, top_k), score_threshold=score_threshold,
                            start_date=start, end_date=end)

        full_ctx = _format_context_blocks(hits)

        if strict_mode and not _has_context(full_ctx):
            st.warning("No matching context; Strict mode is active.")
            st.markdown("### Year-by-Year Talking Points")
            st.code("Not found in corpus.")
            st.markdown("### Messaging Evolution (Narrative)")
            st.code("Not found in corpus.")
        else:
            # 1) Year-by-Year talking points
            if strict_mode:
                sys_focus = (
                    "You are a senior IMF communications strategist. "
                    "Using ONLY the provided context grouped by date headers, produce talking points. "
                    "If the context is insufficient for any period, reply exactly: Not found in corpus."
                )
            else:
                sys_focus = (
                    "You are a senior IMF communications strategist. Using ONLY the provided context, "
                    "produce clear, issue-focused talking points by date. Cite with (YYYY-MM-DD — Title)."
                )
            usr_focus = f"Context grouped by date and title:\n\n{full_ctx}\n\n---\n\nTask: Produce concise talking points grouped by date."

            # 2) Narrative of messaging evolution
            if strict_mode:
                sys_evol = (
                    "Analyze how messaging evolved using ONLY the provided context. "
                    "If the context is insufficient, reply exactly: Not found in corpus."
                )
            else:
                sys_evol = (
                    "Analyze how messaging evolved across the entire date range using ONLY the provided context. "
                    "Be concrete about shifts in emphasis; cite with (YYYY-MM-DD — Title) where helpful."
                )
            usr_evol = f"Context:\n\n{full_ctx}\n\n---\n\nTask: Write a concise evolution narrative."

            if not OPENAI_OK or not os.environ.get("OPENAI_API_KEY"):
                st.error("OpenAI client unavailable or OPENAI_API_KEY not set.")
            else:
                client = OpenAI()
                with st.spinner("Generating year-by-year talking points..."):
                    resp1 = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=[{"role": "system", "content": sys_focus},
                                  {"role": "user", "content": usr_focus}],
                        temperature=0.0 if strict_mode else 0.2,
                    )
                    out1 = (resp1.choices[0].message.content or "").strip()

                with st.spinner("Generating evolution narrative..."):
                    resp2 = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=[{"role": "system", "content": sys_evol},
                                  {"role": "user", "content": usr_evol}],
                        temperature=0.0 if strict_mode else 0.2,
                    )
                    out2 = (resp2.choices[0].message.content or "").strip()

                st.markdown("### Year-by-Year Talking Points")
                if strict_mode and out1.lower() == "not found in corpus.":
                    st.code("Not found in corpus.")
                else:
                    st.write(out1)

                st.markdown("### Messaging Evolution (Narrative)")
                if strict_mode and out2.lower() == "not found in corpus.":
                    st.code("Not found in corpus.")
                else:
                    st.write(out2)

# ===== Rapid Response Tab =====
with tab_rr:
    st.subheader("📰 Rapid Response (Media Q&A)")
    rr_q = st.text_area("Paste a media inquiry or topic prompt:", height=120, key="q_rr")
    run_rr = st.button("Draft response", key="run_rr")
    if run_rr and rr_q.strip():
        with st.spinner("Retrieving supporting excerpts..."):
            rr_hits = retrieve(query=rr_q, k=max(12, top_k), score_threshold=score_threshold,
                               start_date=start, end_date=end)
        ctx_rr = _format_context_blocks(rr_hits)

        if strict_mode and not _has_context(ctx_rr):
            st.code("Not found in corpus.")
        else:
            if strict_mode:
                sys = (
                    "You are Kristalina Georgieva's communications aide. "
                    "Using ONLY the provided excerpts from her speeches, answer the inquiry. "
                    "If the excerpts are insufficient, reply exactly: Not found in corpus."
                )
            else:
                sys = (
                    "You are Kristalina Georgieva's communications aide. "
                    "Using the provided excerpts from her speeches, produce 3–5 bullet points "
                    "followed by a short narrative paragraph suitable for media."
                )
            usr = f"Excerpts:\n\n{ctx_rr}\n\n---\n\nInquiry:\n{rr_q}"

            if not OPENAI_OK or not os.environ.get("OPENAI_API_KEY"):
                st.error("OpenAI client unavailable or OPENAI_API_KEY not set.")
            else:
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "system", "content": sys},
                              {"role": "user", "content": usr}],
                    temperature=0.0 if strict_mode else 0.2,
                )
                out = (resp.choices[0].message.content or "").strip()
                if strict_mode and out.lower() == "not found in corpus.":
                    st.code("Not found in corpus.")
                else:
                    st.write(out)

# ===== Draft Assist Tab =====
with tab_draft:
    st.subheader("📝 Draft Assist (Outline)")
    draft_q = st.text_input("Topic or angle for a short outline (speaker's voice)", key="q_draft")
    run_d = st.button("Generate outline", key="run_draft")
    if run_d and draft_q.strip():
        with st.spinner("Retrieving style & substance context..."):
            draft_hits = retrieve(query=draft_q, k=max(16, top_k), score_threshold=score_threshold,
                                  start_date=start, end_date=end)
        ctx = _format_context_blocks(draft_hits)

        if strict_mode and not _has_context(ctx):
            st.code("Not found in corpus.")
        else:
            if strict_mode:
                sys = (
                    "You are drafting a first-pass speech outline in the speaker's established style, "
                    "using ONLY the provided context. If the context is insufficient, reply exactly: Not found in corpus."
                )
            else:
                sys = (
                    "You are drafting a first-pass speech outline in the speaker's established style, "
                    "grounded ONLY in the provided context. Keep it concise and structured."
                )
            usr = f"Context:\n\n{ctx}\n\n---\n\nTask: Draft a concise outline for: {draft_q}"

            if not OPENAI_OK or not os.environ.get("OPENAI_API_KEY"):
                st.error("OpenAI client unavailable or OPENAI_API_KEY not set.")
            else:
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "system", "content": sys},
                              {"role": "user", "content": usr}],
                    temperature=0.0 if strict_mode else 0.2,
                )
                out = (resp.choices[0].message.content or "").strip()
                if strict_mode and out.lower() == "not found in corpus.":
                    st.code("Not found in corpus.")
                else:
                    st.write(out)

# ===== Corpus Tab =====
with tab_corpus:
    st.subheader("📚 Corpus Browser & Maintenance")

    vecs, meta = load_index()
    # Aggregate by source_id
    by_source: Dict[str, Dict[str, Any]] = {}
    for m in meta:
        sid = m.get("source_id", "unknown")
        item = by_source.setdefault(sid, {"source_id": sid, "source_name": m.get("source_name", "?"),
                                          "title": m.get("title"), "date_iso": m.get("date_iso"),
                                          "num_chunks": 0})
        item["num_chunks"] += 1

    if by_source:
        # Display as simple table
        rows = sorted(by_source.values(), key=lambda x: (x.get("date_iso") or "", x.get("source_name") or ""))
        st.write(f"**Documents in corpus:** {len(rows)}")
        # Table-ish markdown
        st.markdown("| Source Name | Date | Title | Chunks | Source ID |")
        st.markdown("|---|---|---|---:|---|")
        for r in rows:
            st.markdown(f"| {r.get('source_name','?')} | {r.get('date_iso','')} | {r.get('title','')} | {r['num_chunks']} | `{r['source_id']}` |")

        # Delete flow
        st.markdown("---")
        st.markdown("### Delete from corpus")
        del_ids = st.text_input("Enter one or more `source_id` values to delete (comma-separated).")
        if st.button("Delete selected"):
            ids = [s.strip() for s in del_ids.split(",") if s.strip()]
            if not ids:
                st.warning("No source_id provided.")
            else:
                # Build mask of rows to keep
                keep_mask = [m.get("source_id") not in ids for m in meta]
                new_meta = [m for m, keep in zip(meta, keep_mask) if keep]
                new_vecs = vecs[keep_mask] if vecs.size and len(keep_mask) == vecs.shape[0] else np.zeros((0, EMBED_DIM), dtype=np.float32)
                save_index(new_vecs, new_meta)
                st.success(f"Deleted items with source_id in: {ids}. Re-run queries to use updated corpus.")
    else:
        st.info("Corpus is empty. Upload content or ingest from PKL.")

    st.markdown("---")
    st.markdown(f"### Re-import from `{PKL_PATH}`")
    if st.button("Import now"):
        try:
            stats = ingest_pkl(PKL_PATH)
            if stats.get("loaded"):
                st.success(f"Imported: {stats['new_docs']} new doc(s), {stats['new_chunks']} new chunk(s).")
            else:
                st.info("No records imported.")
        except Exception as e:
            st.error(f"Import failed: {e}")

# ---- Manual Upload / Ingest ----
st.divider()
st.subheader("📥 Upload / Ingest Speeches (Manual)")
st.write("Upload TXT, MD, PDF, or DOCX files. The app will chunk and index them locally (no web calls).")

uploads = st.file_uploader("Add speeches to corpus", type=["txt", "md", "pdf", "docx"], accept_multiple_files=True)

if uploads:
    ensure_store()
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for f in uploads:
        raw = f.read()
        content = read_file_to_text(f.name, raw)
        if not content.strip():
            st.warning(f"Skipped (empty/unreadable): {f.name}")
            continue
        chunks = chunk_text(content)
        if not chunks:
            st.warning(f"No text chunks found: {f.name}")
            continue

        # In manual uploads, synthesize a stable source_id based on name+content digest
        relname = f"manual/{f.name}"
        digest = file_digest(relname, raw)
        title_guess = os.path.splitext(os.path.basename(f.name))[0].replace("_", " ").replace("-", " ").strip()
        date_guess = try_parse_date_from_text(content) or try_parse_date_from_text(title_guess)

        for i, ch in enumerate(chunks):
            metas.append({
                "id": f"{digest}:{i}",
                "chunk_id": i,
                "source_id": digest,
                "source_name": relname,
                "title": title_guess,
                "date_iso": date_guess,  # may be None
                "text": ch,
            })
            texts.append(ch)

    if texts:
        try:
            client = OpenAI()
            new_vecs = embed_texts(client, texts)
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            st.stop()

        vecs, meta = load_index()
        merged_vecs = new_vecs if vecs.size == 0 else np.vstack([vecs, new_vecs])
        merged_meta = metas if not meta else meta + metas
        save_index(merged_vecs, merged_meta)
        st.success(f"Ingested {len(texts)} chunks from {len(uploads)} file(s).")
    else:
        st.info("No new chunks ingested.")

st.divider()
st.caption("Strict mode = 100% contained: no web calls, no general model knowledge beyond retrieved chunks.")
