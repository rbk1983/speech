# news_utils.py — Google News RSS integration (with optional embedding-based ranking)
import re, time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus
import feedparser
import numpy as np
import os

try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    _client = None

_HTML_TAG = re.compile(r"<[^>]+>")

def _clean_html(s: str) -> str:
    if not s:
        return ""
    return _HTML_TAG.sub("", s).strip()

def google_news_rss_url(topic: str, since_hours: int = 24) -> str:
    q = f"{topic} when:{since_hours}h"
    base = "https://news.google.com/rss/search"
    params = f"?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
    return base + params

def _parse_published(entry):
    # feedparser returns published_parsed (struct_time) sometimes
    if getattr(entry, "published_parsed", None):
        try:
            tt = entry.published_parsed
            return datetime(*tt[:6], tzinfo=timezone.utc).isoformat()
        except Exception:
            pass
    if getattr(entry, "updated_parsed", None):
        try:
            tt = entry.updated_parsed
            return datetime(*tt[:6], tzinfo=timezone.utc).isoformat()
        except Exception:
            pass
    return None

def fetch_news(topic: str, since_hours: int = 24, limit: int = 20):
    """Fetch recent news items for a topic via Google News RSS."""
    url = google_news_rss_url(topic, since_hours=since_hours)
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[: limit * 2]:  # overfetch then trim
        title = _clean_html(getattr(e, "title", ""))
        link = getattr(e, "link", "")
        summary = _clean_html(getattr(e, "summary", ""))
        source = ""
        try:
            if hasattr(e, "source") and getattr(e.source, "title", ""):
                source = e.source.title
        except Exception:
            pass
        published_at = _parse_published(e)
        if not title or not link:
            continue
        items.append({
            "title": title,
            "url": link,
            "summary": summary,
            "source": source,
            "published_at": published_at
        })

    # Deduplicate by URL
    seen, dedup = set(), []
    for it in items:
        if it["url"] in seen:
            continue
        seen.add(it["url"])
        dedup.append(it)
    return dedup[:limit]

def _embed_batch(texts):
    if _client is None:
        return None
    try:
        resp = _client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return [d.embedding for d in resp.data]
    except Exception:
        return None

def _cosine(a, b):
    a = np.array(a, dtype="float32")
    b = np.array(b, dtype="float32")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)

def rank_news_by_query(query: str, items, top_k: int = 10):
    """Rank items by semantic similarity to query using OpenAI embeddings.
       Falls back to naive keyword overlap if embeddings unavailable.
    """
    if not items:
        return []
    # Try embeddings first
    vec_q = _embed_batch([query])
    if vec_q and len(vec_q) == 1:
        vec_items = _embed_batch([f"{it['title']} — {it.get('summary','')}" for it in items])
        if vec_items:
            scored = [(it, _cosine(vec_q[0], v)) for it, v in zip(items, vec_items)]
            scored.sort(key=lambda t: t[1], reverse=True)
            return [it for it,_ in scored[:top_k]]

    # Fallback: keyword overlap
    qset = set(query.lower().split())
    scored = []
    for it in items:
        text = f"{it['title']} {it.get('summary','')}".lower()
        overlap = sum(1 for w in qset if w in text)
        scored.append((it, overlap))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [it for it,_ in scored[:top_k]]
