# rag_utils.py — precision retrieval: speech-level scoring, strictness, diversification, LLM rerank
import os, json, datetime as _dt, hashlib, re
from typing import List, Tuple, Dict, Any
import numpy as np
import faiss
import pandas as pd
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    if "title" not in df.columns: df["title"] = "(untitled)"
    if "link" not in df.columns: df["link"] = ""
    return df

def load_index():
    if not os.path.exists("index.faiss"):
        raise FileNotFoundError("index.faiss not found (build index).")
    index = faiss.read_index("index.faiss")
    with open("meta.json","r",encoding="utf-8") as f:
        meta_raw = json.load(f)
    metas = meta_raw.get("metas", meta_raw)
    with open("chunks.json","r",encoding="utf-8") as f:
        chunks_raw = json.load(f)
    chunks = chunks_raw.get("chunks", chunks_raw)
    return index, metas, chunks

def _passes_date(meta, filters):
    if not filters: return True
    try:
        d = _dt.date.fromisoformat(str(meta.get("date")))
    except Exception:
        return False
    if filters.get("date_from") and d < filters["date_from"]: return False
    if filters.get("date_to") and d > filters["date_to"]: return False
    return True

def _exact_phrase_in(text: str, phrase: str) -> bool:
    if not phrase: return True
    if '"' in phrase:
        # pull quoted phrases
        qs = re.findall(r'"([^"]+)"', phrase)
        if qs:
            return all(q.lower() in text.lower() for q in qs)
    return phrase.lower() in text.lower()

def _contains_excluded(text: str, excludes: List[str]) -> bool:
    tl = text.lower()
    return any(x.lower() in tl for x in excludes)

def _date_key(meta_date):
    try: return _dt.datetime.fromisoformat(str(meta_date))
    except Exception: return _dt.datetime.min

# speech key
def _skey(m):
    return (m.get("title"), m.get("date"), m.get("link"))

def _embed(texts: List[str], model="text-embedding-3-large") -> List[List[float]]:
    if not texts: return []
    res = _client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in res.data]

def _mmr(items, query_vec, cand_vecs, lambda_mult=0.5, top_k=25):
    # items: list of dicts with 'skey'
    selected = []
    selected_ix = set()
    cand_ix = list(range(len(items)))
    if not cand_vecs: return items[:top_k]
    import numpy as np
    q = np.array(query_vec, dtype="float32")
    C = np.array(cand_vecs, dtype="float32")
    # normalize
    q = q / (np.linalg.norm(q)+1e-9)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True)+1e-9)
    sim_q = Cn @ q
    sim_mat = Cn @ Cn.T
    while len(selected) < min(top_k, len(items)) and cand_ix:
        if not selected:
            i = int(np.argmax(sim_q[cand_ix]))
            pick = cand_ix[i]
        else:
            max_div = -1e9; pick = cand_ix[0]
            for j in cand_ix:
                diversity = min(sim_mat[j, s] for s in selected_ix) if selected_ix else 0.0
                score = lambda_mult*sim_q[j] - (1-lambda_mult)*diversity
                if score > max_div:
                    max_div, pick = score, j
        selected.append(items[pick]); selected_ix.add(pick); cand_ix.remove(pick)
    return selected

def _core_topics_cache_path():
    return "core_topics.json"

def _get_core_topics(skey, meta, chunk_text, model) -> List[str]:
    path = _core_topics_cache_path()
    db = {}
    if os.path.exists(path):
        try:
            db = json.load(open(path,"r",encoding="utf-8"))
        except Exception:
            db = {}
    key = f"{skey[0]}|{skey[1]}|{skey[2]}"
    if key in db:
        return db[key]
    # generate
    sys = "Identify the top 2 core topics/themes of this speech excerpt as short phrases (2–4 words)."
    usr = f"Title: {meta.get('title')}\nDate: {meta.get('date')}\nExcerpt:\n{chunk_text[:1200]}\n\nReturn as a comma-separated list, max 2."
    try:
        resp = _client.chat.completions.create(
            model=("gpt-4o-mini"),
            messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
            temperature=0.2,
            max_tokens=60
        )
        txt = resp.choices[0].message.content.strip()
        topics = [t.strip() for t in re.split(r"[,\n]", txt) if t.strip()][:2]
    except Exception:
        topics = []
    db[key] = topics
    try:
        json.dump(db, open(path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:
        pass
    return topics

def retrieve_speeches(query: str, index, metas, chunks, filters=None, sort="newest",
                      precision=True, strictness=0.6, exact_phrase=True, exclude_terms=None,
                      core_topic_only=False, use_llm_rerank=True, model="gpt-4o-mini"):
    """
    Returns (speeches, total_before) where speeches is a list of dicts:
      {"meta": m, "best_chunk": ch, "best_idx": i, "score": s}
    """
    exclude_terms = exclude_terms or []
    # query embedding
    qv = _client.embeddings.create(model="text-embedding-3-large", input=query).data[0].embedding
    # search more than needed
    D, I = index.search(np.array([qv], dtype="float32"), 400)
    n = len(metas)
    cands = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= n: continue
        m = metas[idx]
        if not _passes_date(m, filters): continue
        txt = chunks[idx] or ""
        if exact_phrase and ('"' in query or len(query.split())>1):
            phrase = query
            if not _exact_phrase_in(txt, phrase): 
                continue
        if exclude_terms and _contains_excluded(txt, exclude_terms):
            continue
        cands.append((idx, m, txt, float(score)))
    total_before = len(cands)

    # group by speech
    group: Dict[Any, List[Tuple[int, Dict, str, float]]] = {}
    for idx, m, txt, s in cands:
        group.setdefault(_skey(m), []).append((idx, m, txt, s))

    speeches = []
    for sk, lst in group.items():
        lst_sorted = sorted(lst, key=lambda t: t[3], reverse=True)
        best_idx, m, best_txt, best_s = lst_sorted[0]
        mean_s = float(np.mean([t[3] for t in lst_sorted]))
        # off-topic penalty: proportion of chunks in speech that matched
        ratio = len(lst_sorted) / max(1, len(lst))
        penalty = 0.0 if ratio >= 0.5 else (0.5 - ratio)  # penalize shallow mentions
        combined = 0.7*best_s + 0.3*mean_s - penalty
        speeches.append({"meta": m, "best_chunk": best_txt, "best_idx": best_idx, "score": combined})

    if not speeches:
        return [], total_before

    # Normalize scores 0..1
    scores = np.array([s["score"] for s in speeches], dtype="float32")
    mn, mx = float(scores.min()), float(scores.max())
    if mx > mn:
        for s in speeches:
            s["score"] = (s["score"] - mn) / (mx - mn)
    else:
        for s in speeches:
            s["score"] = 0.5

    # Core-topic-only filter (on top 40 to save calls)
    if core_topic_only:
        top_for_tag = sorted(speeches, key=lambda x: x["score"], reverse=True)[:40]
        allowed_keys = set()
        q_low = query.lower()
        for sp in top_for_tag:
            topics = _get_core_topics(_skey(sp["meta"]), sp["meta"], sp["best_chunk"], model=model)
            if any(t and t.lower() in q_low or q_low in t.lower() for t in topics):
                allowed_keys.add(_skey(sp["meta"]))
        speeches = [sp for sp in speeches if _skey(sp["meta"]) in allowed_keys]

    # Apply threshold
    speeches = [sp for sp in speeches if sp["score"] >= strictness]

    # Sort
    if sort == "newest":
        speeches.sort(key=lambda sp: _date_key(sp["meta"].get("date","")), reverse=True)
    else:
        speeches.sort(key=lambda sp: sp["score"], reverse=True)

    # Diversify via MMR on top 50
    top = speeches[:50]
    # candidate vectors = embed title + first 200 chars of best chunk
    texts = [ (sp["meta"].get("title","") + " " + (sp["best_chunk"][:200] or "")) for sp in top ]
    try:
        cand_vecs = _embed(texts)
        top = _mmr(top, qv, cand_vecs, lambda_mult=0.6, top_k=min(25, len(top)))
        # prepend diversified top, then the rest
        speeches = top + [sp for sp in speeches if sp not in top]
    except Exception:
        pass

    # Optional LLM rerank on first 25
    if use_llm_rerank and speeches:
        head = speeches[:25]
        sys = "Rank speeches by how centrally they address the user's topic, based ONLY on titles and excerpts. Return CSV: rank,title,date,score"
        usr_lines = [f"- {sp['meta'].get('date')} — {sp['meta'].get('title')}: {sp['best_chunk'][:350].replace('\\n',' ')}" for sp in head]
        usr = f"Topic: {query}\nSpeeches:\n" + "\n".join(usr_lines)
        try:
            resp = _client.chat.completions.create(
                model=("gpt-4o-mini"),
                messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
                temperature=0.2, max_tokens=250
            )
            text = resp.choices[0].message.content.strip()
            import pandas as pd, io
            df = pd.read_csv(io.StringIO(text))
            # Map back by title+date
            rank_map = {}
            for _, r in df.iterrows():
                key = (str(r.get("title","")).strip(), str(r.get("date","")).strip())
                rank_map[key] = float(r.get("score", 0))
            def rr(sp):
                k = (str(sp["meta"].get("title","")).strip(), str(sp["meta"].get("date","")).strip())
                return rank_map.get(k, 0)
            head_sorted = sorted(head, key=lambda sp: rr(sp), reverse=True)
            speeches = head_sorted + [sp for sp in speeches if sp not in head]
        except Exception:
            pass

    return speeches, total_before

def sources_from_speeches(items):
    seen, out = set(), []
    for sp in items:
        m = sp["meta"]
        key = (m.get("date"), m.get("title"), m.get("link"))
        if key in seen: continue
        seen.add(key); out.append(key)
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
        excerpt = textwrap.shorten((ch or '').replace("\\n"," "), width=char_limit, placeholder="…")
        out.append(head + "\\n" + excerpt)
    return "\\n\\n".join(out) if out else "(no matching context)"

def llm(system_prompt, user_prompt, model="gpt-4o-mini", max_tokens=700, temperature=0.3):
    def _call(m):
        resp = _client.chat.completions.create(
            model=m,
            messages=[{"role":"system","content":system_prompt.strip()},
                      {"role":"user","content":user_prompt.strip()}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    try:
        return _call(model)
    except Exception:
        for fb in ("gpt-4o-mini","gpt-4o"):
            try: return _call(fb)
            except Exception: continue
        raise


def llm_stream(system_prompt, user_prompt, model="gpt-4o-mini", max_tokens=700, temperature=0.3):
    """Streamed variant of ``llm`` returning (generator, model_used)."""

    def _stream_from(resp):
        for chunk in resp:
            txt = chunk.choices[0].delta.content or ""
            if txt:
                yield txt

    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        return _stream_from(resp), model
    except Exception:
        for fb in ("gpt-4o-mini", "gpt-4o"):
            try:
                resp = _client.chat.completions.create(
                    model=fb,
                    messages=[
                        {"role": "system", "content": system_prompt.strip()},
                        {"role": "user", "content": user_prompt.strip()},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                return _stream_from(resp), fb
            except Exception:
                continue
        raise
