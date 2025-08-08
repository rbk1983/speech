# phase3_utils.py — Messaging trajectory + alignment score
import hashlib, io
import numpy as np
import pandas as pd
from rag_utils import llm

# ---------- Helper: robust CSV -> DataFrame ----------
def _to_df(csv_text, cols):
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        # try to coerce simple "issue,score" lines
        rows = []
        for line in csv_text.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= len(cols):
                rows.append(parts[:len(cols)])
        df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ---------- Extract per-year issues time series ----------
def issues_time_series(topic, context_md, model):
    """
    Ask the LLM to produce year-level issue weights from the provided context.
    Returns DataFrame columns: year, issue, score (0..1).
    """
    sys = ("You analyze the provided citations and produce a year-by-year breakdown of ISSUE emphasis for the user's topic. "
           "Issues should be short, human-meaningful (e.g., 'Climate finance', 'Debt sustainability', 'AI & productivity'). "
           "Use ONLY the context; infer weights 0..1 by how much the issue is emphasized.")
    usr = f"""Topic: {topic}
Context (each item starts with [YYYY-MM-DD — Title](link)):
{context_md}

Return CSV columns: year,issue,score
- year = YYYY
- issue = concise issue name
- score = 0..1 emphasis in that year for this topic
Include 5–30 rows total."""
    key = f"ts::{hashlib.sha256((topic+context_md).encode()).hexdigest()}::{model}"
    text = llm(sys, usr, model=model, max_tokens=600, temperature=0.2)
    text = text[0] if isinstance(text, tuple) else text
    df = _to_df(text, ["year","issue","score"])
    # clean
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).clip(0,1)
    df["issue"] = df["issue"].astype(str).str.strip()
    df = df.dropna(subset=["year","issue"])
    return df

# ---------- Forecast next period via simple linear trend per issue ----------
def forecast_issues(ts_df):
    if ts_df is None or ts_df.empty:
        return pd.DataFrame()
    out = []
    max_year = int(ts_df["year"].dropna().max())
    min_year = int(ts_df["year"].dropna().min())
    nxt = max_year + 1
    for issue, grp in ts_df.groupby("issue"):
        g = grp.sort_values("year")
        # need at least 2 points
        if g["year"].nunique() < 2:
            continue
        x = g["year"].astype(float).values
        y = g["score"].astype(float).values
        # linear fit
        try:
            m, b = np.polyfit(x, y, 1)
            y_next = float(np.clip(m * nxt + b, 0, 1))
        except Exception:
            continue
        # actual rows
        for yy, ss in zip(x, y):
            out.append({"year": int(yy), "issue": issue, "score": float(ss), "kind": "actual"})
        out.append({"year": int(nxt), "issue": issue, "score": y_next, "kind": "forecast"})
    return pd.DataFrame(out)

# ---------- Trajectory narrative ----------
def trajectory_narrative(topic, ts_df, forecast_df, model):
    sys = ("You are a senior communications strategist. Using ONLY the time series (and forecasts if present), "
           "write a short narrative on messaging trajectory for this topic: what's rising, what's fading, and why it matters. "
           "Be concrete and media-ready. Avoid speculation beyond the data.")
    ts_csv = ts_df.to_csv(index=False) if isinstance(ts_df, pd.DataFrame) and not ts_df.empty else "(none)"
    fc_csv = forecast_df.to_csv(index=False) if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty else "(none)"
    usr = f"""Topic: {topic}
Time series (CSV):
{ts_csv}

Forecast (CSV):
{fc_csv}

Return 6–10 sentences, crisp, with clear takeaways and suggested emphasis for next quarter."""
    text = llm(sys, usr, model=model, max_tokens=500, temperature=0.25)
    return text

# ---------- Alignment score (numeric) ----------
def alignment_score_value(topic, playbook, context_md, model):
    """
    Produce a single 0..100 alignment score against playbook issues.
    Uses LLM to map issues in context to playbook keys and compare coverage to targets.
    """
    issues = list((playbook.get("issues") or {}).keys())
    if not issues:
        return None
    targets = {k: float(v.get("target", 0.7)) for k, v in (playbook.get("issues") or {}).items()}
    sys = ("Score alignment (0..100) between the provided excerpts and the playbook issues. "
           "Interpret alignment as how closely the emphasis matches the target proportions across issues. "
           "Return only the integer score.")
    # Provide targets as CSV for clarity
    import io as _io
    tgt_csv = "issue,target\n" + "\n".join([f"{k},{targets[k]}" for k in targets])
    usr = f"""Playbook targets (CSV):
{tgt_csv}

Context:
{context_md}

Return only one integer: the alignment score 0..100 (no extra text)."""
    try:
        text = llm(sys, usr, model=model, max_tokens=20, temperature=0.0)
        text = text[0] if isinstance(text, tuple) else text
        # extract first integer
        import re
        m = re.search(r'(\d{1,3})', str(text))
        if not m:
            return None
        val = int(m.group(1))
        return int(max(0, min(100, val)))
    except Exception:
        return None
