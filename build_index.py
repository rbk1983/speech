import pandas as pd
import faiss
import numpy as np
import json
from tqdm import tqdm
from rag_utils import embed_texts

# Load speeches data
df = pd.read_pickle("speeches_data.pkl")

# Split into chunks
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = []
meta = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    for ch in chunk_text(row['transcript']):
        chunks.append(ch)
        meta.append({
            'title': row['title'],
            'date': str(row['date']),
            'link': row['link']
        })

# Embed
embeddings = embed_texts(chunks)
dim = embeddings.shape[1]

# Save FAISS index
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, "index.faiss")
json.dump(meta, open("meta.json", "w"))
json.dump(chunks, open("chunks.json", "w"))

print("Index built with", len(chunks), "chunks")
