# rag_utils.py — retrieval and LLM helpers

import os
import json
import pickle
import pandas as pd
import faiss
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Loaders
def load_df():
    """Load speeches dataframe from pickle."""
    if os.path.exists("speeches_data_reclassified.pkl"):
        return pd.read_pickle("speeches_data_reclassified.pkl")
    elif os.path.exists("speeches_data.pkl"):
        return pd.read_pickle("speeches_data.pkl")
    else:
        raise FileNotFoundError("No speeches_data.pkl or speeches_data_reclassified.pkl found in repo.")

def load_index():
    """Load FAISS index, metadata, and chunks."""
    if not os.path.exists("index.faiss"):
        raise FileNotFoundError("index.faiss not found.")
    index = faiss.read_index("index.faiss")
    with open("meta.json", "r", encoding="utf-8") as f:
        metas = json.load(f)
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, metas, chunks

# --- Retriever
def retrieve(query, index, metas, chunks, k=16, filters=None):
    """Search FAISS, dedupe to one chunk per speech, then sort by date (newest first)."""
    from openai import OpenAI
    import numpy as np
    import datetime as _dt

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Embed query
    qv = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    D, I = index.search(np.array([qv], dtype="float32"), k * 4)  # over-fetch, then filter/dedupe
    candidates = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        m = metas[idx]
        if filters and not _passes(m, filters):
            continue
        candidates.append((idx, m, chunks[idx], float(score)))

    # Dedupe to one hit per speech (title+date+link) keeping the highest-scoring one
    best = {}
    for idx, m, ch, score in candidates:
        key = (m.get("title"), m.get("date"), m.get("link"))
        if key not in best or score > best[key][3]:
            best[key] = (idx, m, ch, score)

    deduped = list(best.values())

    # Sort by date desc (expects 'YYYY-MM-DD' or ISO-like). If anything weird, fall back safely.
    def _date_key(meta_date):
        try:
            return _dt.datetime.fromisoformat(str(meta_date))
        except Exception:
            return _dt.datetime.min

    deduped.sort(key=lambda t: _date_key(t[1].get("date", "")), reverse=True)

    # Return the top-k without the score
    return [(idx, m, ch) for (idx, m, ch, _score) in deduped[:k]]


# --- LLM wrapper
def llm(system_prompt, user_prompt, model="gpt-4o-mini", max_tokens=800):
    """Query OpenAI ChatCompletion."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content
