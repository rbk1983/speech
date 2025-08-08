# rag_utils.py — robust RAG helpers + sorting, filters, pagination
import os, json, datetime as _dt
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI

# One client
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Loaders ----------
def load_df():
    if os.path.exists("speeches_data_reclassified.pkl"):
        df = pd.read_pickle("speeches_data_reclassified.pkl")
    elif os.path.exists("speeches_data.pkl"):
        df = pd.read_pickle("speeches_data.pkl")
    else:
        raise FileNotFoundError("Add speeches_data_reclassified.pkl or speeches_data.pkl to repo root.")
    if "date" not in df.columns or "transcript" not in df.columns:
        raise ValueError("Dataset needs 'date' and 'transcript' columns.")
    df["date"] = pd.to_datetime(df["date"])
    # normalize helpers
    if "title" not in df.columns: df["title"] = "(untitled)"
    if "link" not in df.columns: df["link"] = ""
    if "themes" not in df.columns and "new_themes" not in df.columns:
        df["themes"] = [[] for _ in range(len(df))]
    return df

def load_index():
    if not os.path.exists("index.faiss"):
        raise FileNotFoundError("index.faiss not found (build in-app).")
    index = faiss.read_index("index.faiss")
    with open("meta.json", "r", encoding="utf-8") as f:
        meta_raw = json.load(f)
    metas = meta_raw.get("metas", meta_raw)
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks_raw = json.load(f)
    chunks = chunks_raw.get("chunks", chunks_raw)
    if not isinstance(metas, list) or not isinstance(chunks, list):
        raise ValueError("meta.json / chunks.json malformed.")
    return index, metas, chunks

# ---------- Filters ----------
def _passes(meta, filters):
    if not filters: return True
    # theme facet
    if filters.get("themes"):
        speech_themes = meta.get("themes") or []
        if not any(t in speech_themes for t in filters["themes"]):
            return False
    # date range facet
    if filters.get("date_from") or filters.get("date_to"):
        try:
            d = _dt.date.fromisoformat(str(meta.get("date")))
        except Exception:
            return False
        if filters.get("date_from") and d < filters["date_from"]:
            return False
        if filters.get("date_to") and d > filters["date_to"]:
            return False
    return True

def _date_key(meta_date):
    try: return _dt.datetime.fromisoformat(str(meta_date))
    except Exception: return _dt.datetime.min

# ---------- Retrieval (relevance or newest) + pagination ----------
def retrieve(query, index, metas, chunks, k=50, filters=None, sort="relevance", offset=0, limit=20):
    """
    sort: 'relevance' or 'newest'
    pagination: offset + limit applied after dedupe/sort
    """
    # Embed with SAME model used in build_index.py
    qv = _client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding

    # over-fetch then filter/dedupe
    import numpy as np
    D, I = index.search(np.array([qv], dtype="float32"), max(k * 8, 50))
    n = len(metas)
    cands = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= n: continue
        m = metas[idx]
        if not _passes(m, filters): continue
        cands.append((idx, m, chunks[idx], float(score)))

    # dedupe to one best chunk per speech (title, date, link)
    best = {}
    for idx, m, ch, score in cands:
        key = (m.get("title"), m.get("date"), m.get("link"))
        if key not in best or score > best[key][3]:
            best[key] = (idx, m, ch, score)
    items = list(best.values())

    # sort
    if sort == "newest":
        items.sort(key=lambda t: _date_key(t[1].get("date","")), reverse=True)
    else:  # relevance
        items.sort(key=lambda t: t[3], reverse=True)

    # paginate
    page = items[offset: offset + limit]
    results = [(i, m, ch) for (i, m, ch, _s) in page]
    total = len(items)
    return results, total

# ---------- Helpers ----------
def sources_from_hits(hits):
    """Return a deduped list of (date, title, link) from hits."""
    seen, out = set(), []
    for _, m, _ in hits:
        key = (m.get("date"), m.get("title"), m.get("link"))
        if key in seen: continue
        seen.add(key)
        out.append(key)
    # newest first
    out.sort(key=lambda t: _date_key(t[0]), reverse=True)
    return out

def format_hits_for_context(hits, limit=12, char_limit=900):
    import textwrap
    out, seen = [], set()
    for (_, m, ch) in hits:
        key = (m.get("date"), m.get("title"))
        if key in seen: continue
        seen.add(key)
        if len(out) >= limit: break
        head = f"[{m.get('date')} — {m.get('title')}]({m.get('link')})"
        excerpt = textwrap.shorten((ch or '').replace("\n"," "), width=char_limit, placeholder="…")
        out.append(head + "\n" + excerpt)
    return "\n\n".join(out) if out else "(no matching context)"

def llm(system_prompt, user_prompt, model="gpt-4o-mini", max_tokens=900, temperature=0.3):
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()
