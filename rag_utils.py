import os
import numpy as np
import openai
import faiss
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

def embed_texts(texts, model="text-embedding-3-small"):
    resp = openai.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data]).astype('float32')

def retrieve(query, k=10):
    index = faiss.read_index("index.faiss")
    meta = json.load(open("meta.json"))
    chunks = json.load(open("chunks.json"))
    q_emb = embed_texts([query])
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        results.append({**meta[idx], "chunk": chunks[idx]})
    return results

def llm_complete(prompt, model="gpt-4o-mini", temp=0.4, max_tokens=500):
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are an expert IMF speech analyst."},
                  {"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()
