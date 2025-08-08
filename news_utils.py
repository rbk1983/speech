# news_utils.py — Google News RSS multi-topic fetch + semantic ranking
import re, time, math, html
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict
import feedparser
import hashlib

try:
    from openai import OpenAI
    import os
    _OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    _client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None
except Exception:
    _client = None
    _OPENAI_KEY = None

# -------------- Query parsing ----------------
def _split_query_terms(q: str, max_terms: int = 5) -> List[str]:
    if not q:
        return []
    q = q.strip()
    # Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', q)
    # Remove them from the string
    rest = re.sub(r'"[^"]+"', ' ', q)
    # Split by commas and "and"
    parts = re.split(r'[,\s]+and[,\s]+|,|\s+', rest)
    parts = [p.strip() for p in parts if p and p.strip().lower() not in {"and","or"}]
    terms = quoted + parts
    # Dedup, keep order
    seen = set()
    out = []
    for t in terms:
        t = re.sub(r'\s+', ' ', t).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
        if len(out) >= max_terms:
            break
    return out or [q]

# -------------- Google RSS fetch ----------------
def _rss_url(term: str, hours: int) -> str:
    # when:Nd filter; cap to 7 days for RSS
    days = max(1, min(7, math.ceil(hours/24)))
    from urllib.parse import quote_plus
    return f"https://news.google.com/rss/search?q={quote_plus(term)}+when%3A{days}d&hl=en-US&gl=US&ceid=US:en"

def _parse_time(entry) -> datetime:
    for k in ("published_parsed","updated_parsed"):
        dt = entry.get(k)
        if dt:
            return datetime(*dt[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)

def _norm_item(entry: dict) -> dict:
    title = html.unescape(entry.get("title","")).strip()
    link = entry.get("link","")
    source = ""
    if "source" in entry and hasattr(entry["source"], "title"):
        source = str(entry["source"].title)
    summary = html.unescape(re.sub("<.*?>","", entry.get("summary",""))).strip()
    published_at = _parse_time(entry).isoformat()
    return {
        "title": title,
        "url": link,
        "source": source,
        "summary": summary,
        "published_at": published_at,
    }

def fetch_news(query: str, since_hours: int = 24, limit: int = 30, return_terms: bool = False):
    """
    Fetch multi-topic Google News RSS results.
    Returns a list of normalized items and (optionally) the parsed sub-queries used.
    Each item: {title,url,source,summary,published_at,score}
    """
    terms = _split_query_terms(query, max_terms=5)
    items: Dict[str, dict] = {}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    for term in terms:
        url = _rss_url(term, since_hours)
        feed = feedparser.parse(url)
        for e in feed.entries[:50]:
            it = _norm_item(e)
            # time filter
            try:
                ts = datetime.fromisoformat(it["published_at"])
            except Exception:
                ts = datetime.now(timezone.utc)
            if ts < cutoff:
                continue
            key = hashlib.sha256(it["url"].encode("utf-8")).hexdigest() if it["url"] else hashlib.sha256((it["title"]+it["source"]).encode("utf-8")).hexdigest()
            if key not in items:
                items[key] = it

    results = list(items.values())

    # Scoring
    for it in results:
        it["score"] = 0.0

    # Semantic ranking if embeddings available
    if _client is not None:
        try:
            q_emb = _client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
            # batch item embeddings
            texts = [f"{it['title']} — {it['summary']}" for it in results]
            if texts:
                embs = _client.embeddings.create(model="text-embedding-3-small", input=texts).data
                import numpy as np
                qv = np.array(q_emb, dtype="float32")
                for it, d in zip(results, embs):
                    v = np.array(d.embedding, dtype="float32")
                    # cosine similarity
                    num = float((qv * v).sum())
                    den = float(( (qv*qv).sum()**0.5 ) * ( (v*v).sum()**0.5 ) + 1e-8)
                    it["score"] = num/den
        except Exception:
            # fallback to keyword overlap
            pass

    if not any(it["score"] for it in results):
        # simple keyword overlap scoring
        q_terms = set([t.lower() for t in _split_query_terms(query, max_terms=10)])
        for it in results:
            t = f"{it['title']} {it['summary']}".lower()
            it["score"] = sum(1 for term in q_terms if term in t)

    # sort & limit
    results.sort(key=lambda x: (x.get("score",0), x.get("published_at","")), reverse=True)
    results = results[:limit]

    if return_terms:
        return results, terms
    return results
