# kristalina_app.py
# --------------------------------------------------
# Requirements (install before running):
#   pip install streamlit openai faiss-cpu pandas tiktoken
# --------------------------------------------------

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import faiss
import tiktoken
from openai import OpenAI

# ---------- Configuration ----------
DATA_PATH   = Path("speeches_data.pkl")
INDEX_PATH  = Path("index.faiss")
META_PATH   = Path("meta.json")
CHUNKS_PATH = Path("chunks.json")
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL  = "gpt-4o-mini"
CHUNK_TOKENS = 450
OVERLAP_TOKENS = 80

# ---------- Helpers ----------
def chunk_text(text: str, enc, size=CHUNK_TOKENS, overlap=OVERLAP_TOKENS):
    ids = enc.encode(text or "")
    chunks = []
    for i in range(0, len(ids), size - overlap):
        sub = ids[i:i+size]
        chunks.append(enc.decode(sub))
    return chunks

def build_index(df: pd.DataFrame):
    enc = tiktoken.get_encoding("cl100k_base")
    client = OpenAI()

    texts, metas = [], []
    for i, r in df.iterrows():
        meta = {
            "speech_id": int(i),
            "title": r.get("title", "(untitled)"),
            "date": str(pd.to_datetime(r["date"]).date()),
            "year": int(pd.to_datetime(r["date"]).year),
            "link": r.get("link", ""),
            "themes": r.get("themes", []) or [],
        }
        for ch in chunk_text(r.get("transcript", ""), enc):
            texts.append(ch)
            metas.append(meta)

    # embed
    vecs = []
    BATCH = 128
    for j in range(0, len(texts), BATCH):
        batch = texts[j:j+BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    vecs = np.array(vecs, dtype="float32")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps({"metas": metas}, indent=2))
    CHUNKS_PATH.write_text(json.dumps({"chunks": texts}, indent=2))

def load_index():
    index = faiss.read_index(str(INDEX_PATH))
    metas = json.loads(META_PATH.read_text())["metas"]
    chunks = json.loads(CHUNKS_PATH.read_text())["chunks"]
    return index, metas, chunks

def ensure_index():
    if INDEX_PATH.exists() and META_PATH.exists() and CHUNKS_PATH.exists():
        return
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Place your dataset in the repo root."
        )
    df = pd.read_pickle(DATA_PATH)
    if "date" not in df.columns or "transcript" not in df.columns:
        raise ValueError("Dataset must contain 'date' and 'transcript' columns.")
    df["date"] = pd.to_datetime(df["date"])
    build_index(df)

def search(query: str, k=5):
    client = OpenAI()
    index, metas, chunks = load_index()
    q_vec = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    q_vec = np.array(q_vec, dtype="float32")[None, :]
    faiss.normalize_L2(q_vec)
    scores, idxs = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        meta = metas[idx].copy()
        meta["snippet"] = chunks[idx]
        meta["score"] = float(score)
        results.append(meta)
    return results

def summarize(text: str, instr: str):
    client = OpenAI()
    prompt = f"{instr}\n\n{text}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def compare_years(year_a: int, year_b: int):
    _, metas, chunks = load_index()
    text_a = "\n\n".join(
        c for c, m in zip(chunks, metas) if m["year"] == year_a
    )
    text_b = "\n\n".join(
        c for c, m in zip(chunks, metas) if m["year"] == year_b
    )
    return summarize(
        f"Year {year_a}:\n{text_a}\n\nYear {year_b}:\n{text_b}",
        "Compare the two years; highlight shifting themes and messaging."
    )

def extract_quotes(year: int, top_n=5):
    _, metas, chunks = load_index()
    text = "\n\n".join(
        f"{m['date']} {m['title']}\n{c}" for c, m in zip(chunks, metas) if m["year"] == year
    )
    prompt = f"Return {top_n} notable quotes from the text with brief context."
    return summarize(text, prompt)

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="Kristalina Speech Explorer", layout="wide")
    st.title("Kristalina Georgieva Speech Explorer")

    ensure_index()
    index, metas, chunks = load_index()
    df = pd.DataFrame(metas)

    st.header("Search speeches")
    query = st.text_input("Enter a topic or phrase")
    if query:
        results = search(query, k=5)
        for r in results:
            st.subheader(f"{r['title']} ({r['date']})")
            st.write(r["snippet"])
            if r["link"]:
                st.markdown(f"[Source]({r['link']})")
            st.markdown("---")

    st.header("Quick compare narratives")
    years = sorted(df["year"].unique())
    col1, col2 = st.columns(2)
    with col1:
        year_a = st.selectbox("Year A", years, index=max(0, len(years)-2))
    with col2:
        year_b = st.selectbox("Year B", years, index=len(years)-1)
    if st.button("Compare"):
        st.write(compare_years(year_a, year_b))

    st.header("Top quotes")
    q_year = st.selectbox("Year for quotes", years, key="quote_year")
    if st.button("Show quotes"):
        st.write(extract_quotes(q_year, top_n=5))

    st.caption("Built with OpenAI + FAISS on Kristalina Georgieva's speeches.")

if __name__ == "__main__":
    main()
