# build_index.py (robust dataset detection)
import os, json
import pandas as pd
import faiss, numpy as np
from openai import OpenAI
import tiktoken

CHUNK_TOKENS = 450
OVERLAP_TOKENS = 80
INDEX_PATH = "index.faiss"
META_PATH = "meta.json"
CHUNKS_PATH = "chunks.json"

def chunk_text(txt, enc, size=CHUNK_TOKENS, overlap=OVERLAP_TOKENS):
    txt = txt or ""
    ids = enc.encode(txt)
    if not ids:
        return []
    chunks = []
    for i in range(0, len(ids), size - overlap):
        sub = ids[i:i+size]
        chunks.append(enc.decode(sub))
    return chunks

def pick_dataset():
    candidates = ["speeches_data_reclassified.pkl", "speeches_data.pkl"]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "No dataset found. Please place 'speeches_data_reclassified.pkl' or 'speeches_data.pkl' in the repo root."
    )

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in Streamlit Secrets (or env) first.")

    pkl = pick_dataset()
    df = pd.read_pickle(pkl)
    if "date" not in df.columns or "transcript" not in df.columns:
        raise ValueError("Dataset must include at least 'date' and 'transcript' columns.")
    df["date"] = pd.to_datetime(df["date"])

    enc = tiktoken.get_encoding("cl100k_base")
    client = OpenAI(api_key=api_key)

    texts, metas = [], []
    for i, r in df.iterrows():
        meta = {
            "speech_id_
