# rag_utils.py — retrieval and LLM helpers (robust + join_chunks)
import os, json, datetime as _dt
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Loaders ----------

def load_df():
    """Load speeches dataframe from pickle."""
    if os.path.exists("speeches_data_reclassified.pkl"):
        df = pd.read_pickle("speeches_data_reclassified.pkl")
    elif os.path.exists("speeches_data.pkl"):
        df = pd.read_pickle("speeches_data.pkl")
    else:
        raise FileNotFoundError("No dataset found. Add speeches_data_reclassified.pkl or speeches_data.pkl to repo root.")
    if "date" not in df.columns:
        raise ValueError("Dataset must have a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_index():
    """Load FAISS index, metadata, and chunks (supports dict or list file formats)."""
    if not os.path.exists("index.faiss"):
        raise FileNotFoundError("index.faiss not found. Build the index first (use the in-app button).")
    index = faiss.read_index("index.faiss")

    with open("meta.json", "r", encoding="utf-8") as f:
        meta_raw = json.load(f)
    metas = meta_raw.get("metas", meta_raw)  # support {"metas":[...]} or [...]

    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks_raw = json.load(f)
    chunks = chunks_raw.get("chunks", chunks_raw)  # support {"chunks":[...]} or [...]

    if not isinstance(metas, list) or not isinstance(chunks, list):
        raise ValueError("meta.json / chunks.json malformed. Expected lists.")
    return index, metas, chunks

# ---------- Filters helper ----------

def _passes(meta, filters):
    if not filters:
        return True
    if "themes" in filters and filters["themes"]:
        speech_themes = meta.get("themes") or []
        if not any(t in speech_themes for t in filters["themes"]):
            return False
    if "years" in filters and filters["years"]:
        if meta.get("year") not in filters["years"]:
            return False
    if "location" in filters and filters["location"]:
        if filters["location"].lower() not in (meta.get("location", "").lower()):
            return False
    return True

# ---------- Retriever ----------

def retrieve(query, index, metas, chunks, k=16, filters=None):
    """Search FAISS, dedupe to one chunk per speech, sort newest first."""
    qv = _client.embeddings.create(
        model="text-embedding-3-large",  # must match build_index.py
        input=query
    ).data[0].embedding

    D, I = index.search(np.array([qv], dtype="float32"), k * 4)
    candidates = []
    n = len(metas)
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= n:
            continue
        m = metas[idx]
        if not _passes(m, filters):
            continue
        candidates.append((idx, m, chunks[idx], float(score)))

    best = {}
    for idx, m, ch, score in candidates:
        key = (m.get("title"), m.get("date"), m.get("link"))
        if key not in best or score > best[key][3]:
            best[key] = (idx, m, ch, score)

    deduped = list(best.values())

    def _date_key(meta_date):
        try:
            return _dt.datetime.fromisoformat(str(meta_date))
        except Exception:
            return _dt.datetime.min

    deduped.sort(key=lambda t: _date_key(t[1].get("date", "")), reverse=True)

    return [(idx, m, ch) for (idx, m, ch, _score) in deduped[:k]]

# ---------- Chunk joiner (fixes your error) ----------

def join_chunks(chunk_list, limit=10):
    """
    Join multiple chunks of text into a single string for LLM prompts.
    `limit` = max number of chunks to include.
    """
    if not chunk_list:
        return ""
    return "\n\n".join(chunk_list[:limit])

# ---------- LLM wrapper ----------

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
