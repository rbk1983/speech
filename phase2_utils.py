# phase2_utils.py — helpers for Phase 2 features
import hashlib, datetime as _dt
import pandas as pd
import yaml
from rag_utils import llm

# ---------- Load YAML templates ----------
def load_playbook(path="messaging_playbook.yaml"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        # Minimal default
        return {
            "issues": {
                "Climate finance": {"target": 0.9, "proof_points": ["RSF", "mitigation", "adaptation"]},
                "Debt sustainability": {"target": 0.8, "proof_points": ["G20 Common Framework", "DSA"]},
                "AI & productivity": {"target": 0.5, "proof_points": ["governance", "skills", "inclusion"]},
                "Geoeconomic fragmentation": {"target": 0.6, "proof_points": ["trade", "supply chains"]},
            },
            "banned_phrases": ["panic", "collapse", "hopeless"]
        }

def load_stakeholders(path="stakeholders.yaml"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {
            "stakeholders": {
                "Investors": ["market confidence", "liquidity", "risk premium"],
                "EM Policymakers": ["fiscal space", "reforms", "debt restructuring"],
                "EU Policymakers": ["single market", "green transition", "competitiveness"],
                "Civil Society": ["poverty", "social protection", "inclusion"],
                "Youth": ["jobs", "skills", "digital"]
            }
        }

# ---------- Caching wrapper ----------
_cache = {}
def _cache_call(key, fn):
    if key in _cache:
        return _cache[key]
    val = fn()
    _cache[key] = val
    return val

# ---------- Alignment (radar + narrative) ----------
def alignment_radar_data(topic, playbook, context_md, model):
    issues = list((playbook.get("issues") or {}).keys())
    if not issues:
        return pd.DataFrame()
    sys = "You score alignment of provided excerpts to the given issues (0..1). Use only the context."
    usr = f"""Issues: {', '.join(issues)}
Context:
{context_md}
Return a CSV with columns: issue,score  (0..1)."""
    key = f"align_scores::{hashlib.sha256((topic+context_md).encode()).hexdigest()}::{model}"
    def run():
        text = llm(sys, usr, model=model, max_tokens=300, temperature=0.2)
        text = text[0] if isinstance(text, tuple) else text
        try:
            return pd.read_csv(pd.io.common.StringIO(text))
        except Exception:
            # fallback: evenly spread
            return pd.DataFrame({"issue": issues, "score": [0.5]*len(issues)})
    df = _cache_call(key, run)
    # clamp
    if "score" in df.columns:
        df["score"] = df["score"].clip(0,1)
    return df

def alignment_narrative(topic, playbook, context_md, model):
    sys = ("You are a comms strategist. Using ONLY context and playbook issues, write a short analysis of alignment: "
           "what's over/under-emphasized, any banned phrasing, and 3 concrete next-step lines to improve alignment. "
           "Cite with (Month YYYY — Title) when possible from headers.")
    usr = f"""Playbook issues: {', '.join((playbook.get('issues') or {}).keys())}
Banned phrases: {', '.join(playbook.get('banned_phrases', []))}
Context:
{context_md}
Return 6–10 sentences, crisp."""
    key = f"align_nar::{hashlib.sha256((topic+context_md).encode()).hexdigest()}::{model}"
    def run():
        return llm(sys, usr, model=model, max_tokens=450, temperature=0.25)
    text = _cache_call(key, run)
    return text[0] if isinstance(text, tuple) else text

# ---------- Rapid Response ----------
def rapid_response_pack(headline, context_md, model, url_hint=""):
    sys = ("You write rapid-response lines consistent with historic remarks. Use ONLY the context. "
           "Tone: calm, factual, forward-looking, no over-promising.")
    usr = f"""Headline/topic: {headline}
URL note (optional): {url_hint}

Relevant excerpts:
{context_md}

Tasks:
- 3–4 press lines (bulleted), each one sentence.
- 3 policy specifics (bulleted) grounded in context with (Month YYYY — Title).
- 3 suggested reporter Q&As (Q and a short A anchored in context).
Return clean Markdown.
"""
    text = llm(sys, usr, model=model, max_tokens=700, temperature=0.2)
    return text[0] if isinstance(text, tuple) else text

# ---------- Tone ----------
def tone_time_series(topic, context_md, model):
    sys = "Classify tone for each citation in context into one of: confident, urgent, cautious, optimistic, warning. Return CSV rows."
    usr = f"""Context:
{context_md}

Return CSV columns: date,tone,score
- date = YYYY-MM (approx ok)
- tone ∈ [confident, urgent, cautious, optimistic, warning]
- score ∈ [0..1] strength of that tone
Include 6–20 rows total."""
    key = f"tone_ts::{hashlib.sha256((topic+context_md).encode()).hexdigest()}::{model}"
    def run():
        text = llm(sys, usr, model=model, max_tokens=400, temperature=0.2)
        text = text[0] if isinstance(text, tuple) else text
        try:
            df = pd.read_csv(pd.io.common.StringIO(text))
            # basic cleaning
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            if "score" in df.columns:
                df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).clip(0,1)
            if "tone" in df.columns:
                df["tone"] = df["tone"].str.lower().str.strip()
            return df.dropna(subset=["date"])
        except Exception:
            return pd.DataFrame(columns=["date","tone","score"])
    return _cache_call(key, run)

def tone_heatmap_data(topic, context_md, model):
    sys = "Aggregate tone by year. Use same tone set. Return CSV with columns: year,tone,score."
    usr = f"""Context:
{context_md}

Return CSV columns: year,tone,score
- year = YYYY
- tone ∈ [confident, urgent, cautious, optimistic, warning]
- score ∈ [0..1]"""
    key = f"tone_hm::{hashlib.sha256((topic+context_md).encode()).hexdigest()}::{model}"
    def run():
        text = llm(sys, usr, model=model, max_tokens=350, temperature=0.2)
        text = text[0] if isinstance(text, tuple) else text
        try:
            df = pd.read_csv(pd.io.common.StringIO(text))
            if "year" in df.columns:
                df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            if "score" in df.columns:
                df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).clip(0,1)
            if "tone" in df.columns:
                df["tone"] = df["tone"].str.lower().str.strip()
            return df.dropna(subset=["year"])
        except Exception:
            return pd.DataFrame(columns=["year","tone","score"])
    return _cache_call(key, run)

# ---------- Stakeholders ----------
def stakeholder_scores(topic, stakeholders_yaml, context_md, model):
    names = list((stakeholders_yaml.get("stakeholders") or {}).keys())
    if not names:
        return pd.DataFrame(columns=["stakeholder","score"])
    sys = "Score how relevant each stakeholder audience is for the provided excerpts (0..1). Use only the context."
    usr = f"""Stakeholders: {', '.join(names)}
Context:
{context_md}
Return CSV columns: stakeholder,score  (0..1 for relevance)."""
    key = f"stake_scores::{hashlib.sha256((topic+context_md).encode()).hexdigest()}::{model}"
    def run():
        text = llm(sys, usr, model=model, max_tokens=300, temperature=0.2)
        text = text[0] if isinstance(text, tuple) else text
        try:
            df = pd.read_csv(pd.io.common.StringIO(text))
            if "score" in df.columns:
                df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).clip(0,1)
            return df
        except Exception:
            return pd.DataFrame({"stakeholder": names, "score": [0.5]*len(names)})
    return _cache_call(key, run)

def stakeholder_narrative(topic, stakeholders_yaml, context_md, model):
    sys = ("Write a short stakeholder-targeting note: who this content serves best, what proof points resonate, "
           "and what lines to avoid. Use ONLY the context; cite with (Month YYYY — Title) when possible.")
    usr = f"""Stakeholders: {', '.join((stakeholders_yaml.get('stakeholders') or {}).keys())}
Context:
{context_md}
Return 6–8 sentences, crisp."""
    key = f"stake_nar::{hashlib.sha256((topic+context_md).encode()).hexdigest()}::{model}"
    def run():
        return llm(sys, usr, model=model, max_tokens=450, temperature=0.25)
    text = _cache_call(key, run)
    return text[0] if isinstance(text, tuple) else text
