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
def retrieve(query, index, metas, chunks, k=10, filters=None):
    """Search FAISS index and return top matches."""
    from openai import OpenAI
    import numpy as np

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Embed query
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    D, I = index.search(np.array([q_emb], dtype="float32"), k)
    results = []
    for idx in I[0]:
        if idx == -1:
            continue
        m = metas[idx]
        if filters:
            skip = False
            for fkey, fvals in filters.items():
                if not set(m.get(fkey, [])) & set(fvals):
                    skip = True
                    break
            if skip:
                continue
        results.append((idx, m, chunks[idx]))
    return results

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
