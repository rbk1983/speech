# app_llm.py — Ingest & Sync with YouTube/Vimeo fetch + Speaker filtering + Year-by-Year Evolution + Pagination
import os, datetime as _dt, hashlib, math, json, re, tempfile, io
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# --- Utilities with safe fallback for `llm_stream` ---
from rag_utils import (
    load_df, load_index,
    retrieve_speeches, sources_from_speeches,
    format_hits_for_context, llm
)

# Optional helpers that may or may not exist in your environment
try:
    from rag_utils import whisper_transcribe  # optional convenience
except Exception:
    whisper_transcribe = None

try:
    from build_index import main as build_index_main
except Exception:
    build_index_main = None

try:
    import git  # GitPython (optional)
except Exception:
    git = None

try:
    import docx2txt  # optional for .docx -> text
except Exception:
    docx2txt = None

try:
    import pdfminer.high_level as pdfminer_high  # optional for .pdf -> text
except Exception:
    pdfminer_high = None

try:
    from bs4 import BeautifulSoup  # optional for .html -> text
except Exception:
    BeautifulSoup = None

# yt-dlp for YouTube/Vimeo/audio fetch
try:
    import yt_dlp  # type: ignore
    _YTDLP_OK = True
except Exception:
    yt_dlp = None
    _YTDLP_OK = False

# Optional: diarization + speaker embedding
try:
    from pyannote.audio import Pipeline as _PyannotePipeline  # type: ignore
    _PYANNOTE_OK = True
except Exception:
    _PyannotePipeline = None
    _PYANNOTE_OK = False

try:
    import torch  # type: ignore
    from speechbrain.pretrained import EncoderClassifier  # type: ignore
    _SB_OK = True
except Exception:
    EncoderClassifier = None
    _SB_OK = False

# OpenAI Whisper fallback for transcription with timestamps
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

def _safe_slug(s: str) -> str:
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_')
    return s or "unnamed"

def _now_iso():
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# Whisper transcription using either helper or OpenAI API (returns text and optional segments)
def transcribe_bytes(file_bytes: bytes, filename: str):
    """Return dict: {'text': str, 'segments': list|None}"""
    if whisper_transcribe:
        try:
            text = whisper_transcribe(file_bytes, filename)
            return {"text": text, "segments": None}
        except Exception as e:
            raise RuntimeError(f"whisper_transcribe failed: {e}")
    if not _OPENAI_OK or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Transcription unavailable: set OPENAI_API_KEY or add a whisper_transcribe helper.")
    client = OpenAI()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_name = tmp.name
    try:
        with open(tmp_name, "rb") as f:
            resp = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="verbose_json")
        text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
        segments = getattr(resp, "segments", None) or (resp.get("segments") if isinstance(resp, dict) else None)
        return {"text": text or "", "segments": segments}
    finally:
        try: os.remove(tmp_name)
        except Exception: pass

def doc_to_text(path: Path) -> str:
    """Convert supported docs to raw text."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".docx":
        if not docx2txt:
            raise RuntimeError("docx2txt not installed; cannot parse .docx.")
        return docx2txt.process(str(path)) or ""
    if suffix == ".pdf":
        if not pdfminer_high:
            raise RuntimeError("pdfminer.six not installed; cannot parse .pdf.")
        return pdfminer_high.extract_text(str(path)) or ""
    if suffix in (".html", ".htm"):
        if not BeautifulSoup:
            raise RuntimeError("bs4 not installed; cannot parse .html.")
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")
    raise RuntimeError(f"Unsupported document type: {suffix}")

def ytdlp_fetch_audio(url: str, out_dir: Path, *, cookies_bytes: bytes | None = None, proxy: str | None = None) -> dict:
    \"\"\"Download best audio from a URL (YouTube/Vimeo/etc.) using yt-dlp.
    Returns {'path': Path, 'title': str, 'ext': str, 'uploader': str, 'upload_date': 'YYYYMMDD'}
    \"\"\"
    if not _YTDLP_OK:
        raise RuntimeError("yt-dlp not installed. Install with: pip install yt-dlp")
    out_dir.mkdir(parents=True, exist_ok=True)

    cookiefile = None
    if cookies_bytes:
        import tempfile
        cf = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        cf.write(cookies_bytes)
        cf.flush()
        cookiefile = cf.name

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title).200B.%(ext)s"),
        "noprogress": True,
        "quiet": True,
        "geo_bypass": True,
        "nocheckcertificate": True,
        "restrictfilenames": False,
        "ignoreerrors": False,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
        # Harden against transient errors / throttling
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 1,
        # Try alternate player clients (helps with some 403s)
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }
    if cookiefile:
        ydl_opts["cookiefile"] = cookiefile
    if proxy:
        ydl_opts["proxy"] = proxy

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError("yt-dlp failed to extract media info.")
        if "entries" in info and info["entries"]:
            info = next((e for e in info["entries"] if e), None)
            if info is None:
                raise RuntimeError("No playable entries in playlist.")
        title = info.get("title") or "audio"
        upload_date = info.get("upload_date")  # YYYYMMDD
        uploader = info.get("uploader") or info.get("channel")
        # locate resulting mp3
        candidates = list(out_dir.glob(f"{title}*.mp3"))
        if not candidates:
            candidates = sorted(out_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError("Audio file not found after download.")
        path = candidates[0]
        return {"path": path, "title": title, "ext": path.suffix.lstrip("."), "uploader": uploader, "upload_date": upload_date}

# ----- Speaker filtering helpers (optional) -----
def _load_diarization_pipeline():
    """Load pyannote diarization with HF token if available."""
    if not _PYANNOTE_OK:
        raise RuntimeError("pyannote.audio not installed. pip install pyannote.audio")
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HUGGINGFACE_TOKEN to use diarization (pyannote/speaker-diarization).")
    pipeline = _PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    return pipeline

def _load_spk_encoder():
    if not _SB_OK:
        raise RuntimeError("speechbrain not installed. pip install speechbrain torch")
    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    return enc

def _wav_bytes_from_path(path: Path) -> bytes:
    # yt-dlp produced mp3; speechbrain handles multiple formats via torchaudio backend
    with open(path, "rb") as f:
        return f.read()

def _speaker_embed(encoder, wav_path: Path):
    import torchaudio  # type: ignore
    signal, sr = torchaudio.load(str(wav_path))
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    emb = encoder.encode_batch(signal)
    return emb.squeeze().detach().cpu()

def diarize_and_filter_audio(audio_path: Path, transcript, ref_audio_file) -> dict:
    """Return {'text': str, 'segments': list|None} filtered to the speaker most similar to ref audio.
       transcript is dict from transcribe_bytes (needs 'segments' with timestamps).
    """
    # Ensure we have segments with timestamps
    segments = transcript.get("segments")
    if not segments:
        raise RuntimeError("Transcript lacks timestamps; cannot align segments to speakers. Use Whisper with 'verbose_json'.")

    # Diarize
    pipeline = _load_diarization_pipeline()
    diar = pipeline(str(audio_path))
    # Build speaker->time windows
    spk_windows = {}
    for turn, _, speaker in diar.itertracks(yield_label=True):
        spk_windows.setdefault(speaker, []).append((turn.start, turn.end))

    # Build reference embedding
    encoder = _load_spk_encoder()
    # Save reference bytes to temp file to pass through torchaudio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as t:
        t.write(ref_audio_file.getvalue())
        ref_path = Path(t.name)
    try:
        ref_emb = _speaker_embed(encoder, ref_path)
    finally:
        try: os.remove(ref_path)
        except Exception: pass

    # For each diarized speaker, compute embedding by pooling segments (first N seconds)
    import torchaudio  # type: ignore
    sims = {}
    for spk, wins in spk_windows.items():
        # Extract up to ~30 seconds for embedding
        # Load full file once
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
        # Collate samples from windows
        total = int(sr * 30)
        chunks = []
        acc = 0
        for (s,e) in wins:
            s_i = max(0, int(s * sr)); e_i = min(waveform.shape[-1], int(e * sr))
            if e_i > s_i:
                seg = waveform[:, s_i:e_i]
                chunks.append(seg)
                acc += seg.shape[-1]
                if acc >= total: break
        if not chunks: continue
        pooled = torch.cat(chunks, dim=1)
        # Temp save pooled to file-like for encoder reuse
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpw:
            torchaudio.save(tmpw.name, pooled, sr)
            spk_emb = _speaker_embed(encoder, Path(tmpw.name))
        # cosine similarity
        cos = torch.nn.functional.cosine_similarity(ref_emb.unsqueeze(0), spk_emb.unsqueeze(0)).item()
        sims[spk] = cos

    if not sims:
        raise RuntimeError("Could not compute speaker similarities; check audio quality and dependencies.")

    # Pick best-matching diarized speaker
    target_spk = max(sims, key=sims.get)

    # Filter transcript segments whose midpoint falls inside target speaker windows
    def midpoint(seg):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))
        return (start + end) / 2.0

    def in_windows(t, windows):
        for (s, e) in windows:
            if s <= t <= e:
                return True
        return False

    target_windows = spk_windows[target_spk]
    kept = [seg for seg in segments if in_windows(midpoint(seg), target_windows)]
    text = " ".join([seg.get("text","") for seg in kept]).strip()
    return {"text": text, "segments": kept}

# ----- Defaults (kept simple) -----
PER_ITEM_TOKENS = 400   # Longer = richer bullets/snippets; shorter = terser (cheaper)
DEFAULT_MODEL   = "gpt-4o-mini"

# Cache for LLM results
_LLM_CACHE = {}
def llm_cached(key, system_prompt, user_prompt, *, model=DEFAULT_MODEL,
               max_tokens=PER_ITEM_TOKENS, temperature=0.3, stream=False):
    from rag_utils import llm as _llm
    try:
        from rag_utils import llm_stream  # type: ignore
    except Exception:
        llm_stream = None

    if key in _LLM_CACHE:
        text, used_model = _LLM_CACHE[key]
        if stream:
            st.markdown(text)
        return text, used_model
    if stream and llm_stream:
        gen, used_model = llm_stream(system_prompt, user_prompt,
                                     model=model, max_tokens=max_tokens,
                                     temperature=temperature)
        text = st.write_stream(gen)
        out = (text, used_model)
    else:
        try:
            text = _llm(system_prompt, user_prompt, model=model,
                        max_tokens=max_tokens, temperature=temperature)
            out = (text, model)
        except Exception:
            fallback = DEFAULT_MODEL
            text = _llm(system_prompt, user_prompt, model=fallback,
                        max_tokens=max_tokens, temperature=temperature)
            out = (text, fallback)
    _LLM_CACHE[key] = out
    return out

# Phase 3 utils (optional)
try:
    from phase3_utils import (
        build_issue_trajectory, forecast_issue_trends, trajectory_narrative
    )
    _HAS_P3 = True
except Exception:
    _HAS_P3 = False

st.set_page_config(page_title="Kristalina Speech Intelligence (LLM)", layout="wide")

# ---------------- Styling (Sticky top bar) ----------------
st.markdown("""
<style>
.sticky-bar{
  position: sticky; top: 0; z-index: 999;
  padding: .5rem .25rem .6rem .25rem;
  background: white;
  border-bottom: 1px solid rgba(0,0,0,.08);
}
.card{
  border: 1px solid rgba(0,0,0,.08);
  padding: .75rem 1rem; border-radius: 12px; margin-bottom: .75rem;
  background: #ffffff;
  box-shadow: 0 1px 4px rgba(0,0,0,.04);
}
.card h4{ margin: 0 0 .25rem 0; }
.small { font-size: 0.9rem; color: #666; }
code { background: rgba(0,0,0,.04); padding: 0 .25rem; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("Kristalina Georgieva — Speech Intelligence")

# ---------------- Robust loaders with clearer errors ----------------
def _index_files_present():
    return all(os.path.exists(p) for p in ["index.faiss", "meta.json", "chunks.json"])

def _ensure_index():
    if _index_files_present():
        return True
    st.warning("Search index not found. You can build it here once.")
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Missing OpenAI API key. Add it to Streamlit secrets, then rerun."); st.stop()
    if st.button("Build index now"):
        with st.spinner("Building index…"):
            try:
                from build_index import main as build_index_main_local
                build_index_main_local()
                st.success("Index built. Click **Rerun** or refresh."); st.stop()
            except Exception as e:
                st.error(f"Index build failed: {e}")
                st.stop()

_ensure_index()

try:
    df = load_df()
except Exception as e:
    st.error(f"Failed to load metadata dataframe: {e}")
    st.stop()

try:
    index, metas, chunks = load_index()
except Exception as e:
    st.error(f"Failed to load search index: {e}")
    st.stop()

# ---------------- Sticky Top Bar ----------------
with st.container():
    st.markdown('<div class="sticky-bar">', unsafe_allow_html=True)
    top_c1, top_c2, top_c3, top_c4 = st.columns([2.2, 2.2, 1.2, 1.2])
    with top_c1:
        query = st.text_input("Search speeches", "", placeholder="e.g., climate finance, inflation, Ukraine")
    with top_c2:
        # Date range
        try:
            min_date = pd.to_datetime(df["date"], errors="coerce").dropna().min().date()
            max_date = pd.to_datetime(df["date"], errors="coerce").dropna().max().date()
        except Exception:
            min_date = _dt.date(2000,1,1)
            max_date = _dt.date.today()
        date_from = st.date_input("From", min_date, key="from")
        date_to = st.date_input("To", max_date, key="to")
    with top_c3:
        sort = st.selectbox("Sort", ["Newest", "Relevance"], index=0)
        layout_mode = st.selectbox("Layout", ["Dense list", "Compact cards"], index=1)
    with top_c4:
        exact_phrase = st.toggle("Exact phrase", value=True, help="Match the exact phrase; reduces noise for multi-word queries.")
        high_precision = st.toggle("High precision ranking", value=True, help="Better ranking via dense retrieval + reranking + diversity controls.")
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("Mode: **Depth** — returns fewer items per step but with richer context for better summaries.")

# Advanced (collapsed)
with st.expander("Advanced settings", expanded=False):
    strictness = st.slider("Strictness", 0.0, 1.0, 0.6, 0.05,
                           help="Higher = tighter match to your query; lower = looser, more exploratory.")
    core_topic_only = st.toggle("Core topic only", value=False,
                                help="Show speeches where the query is a central theme (not just a passing mention).")
    model_preferred = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)

filters = {"date_from": date_from, "date_to": date_to}
sort_key = "newest" if sort == "Newest" else "relevance"

def _parse_date(meta_date):
    try:
        return pd.to_datetime(str(meta_date), errors="coerce")
    except Exception:
        return pd.NaT

def _enforce_newest_first(items):
    return sorted(items, key=lambda sp: _parse_date(sp["meta"].get("date")), reverse=True)

# Session state for pagination
if "page" not in st.session_state:
    st.session_state.page = 1

def reset_pagination():
    st.session_state.page = 1

speeches, total_before = [], 0
if query and query.strip():
    fp = f"{query}|{date_from}|{date_to}|{sort_key}|{exact_phrase}|{high_precision}|{core_topic_only}|{strictness}"
    if st.session_state.get("fp") != fp:
        st.session_state.fp = fp
        reset_pagination()
    try:
        speeches, total_before = retrieve_speeches(
            query=query,
            index=index, metas=metas, chunks=chunks,
            filters=filters,
            sort=sort_key,
            precision=high_precision,
            strictness=strictness,
            exact_phrase=exact_phrase,
            exclude_terms=[],
            core_topic_only=core_topic_only,
            use_llm_rerank=True,
            model=model_preferred
        )
    except Exception as e:
        st.error(f"Search failed: {e}")
        speeches = []

    if speeches:
        if sort_key == "newest":
            speeches = _enforce_newest_first(speeches)

        st.caption(f"Matched speeches: {len(speeches)} (from {total_before} initial candidates).")
    else:
        st.info("No results. Try loosening strictness or widening the date range.")

else:
    st.info("Enter a search term above to explore speeches.")

# ---------------- Tabs ----------------
tabs = ["Results", "Thematic Evolution", "Briefing Pack", "Rapid Response", "Draft Assist", "Ingest & Sync"]
try:
    from phase3_utils import build_issue_trajectory  # check existence
    tabs.insert(3, "Trajectory")
    _HAS_P3 = True
except Exception:
    _HAS_P3 = False

tab_objs = st.tabs(tabs)

# Map tabs
tab_res  = tab_objs[0]
tab_comp = tab_objs[1]
tab_brief = tab_objs[2]
offset = 0
if _HAS_P3:
    tab_traj = tab_objs[3]; offset = 1
tab_rr   = tab_objs[3+offset]
tab_draft= tab_objs[4+offset]
tab_ing  = tab_objs[5+offset]

def _to_hits(items):
    return [(sp["best_idx"], sp["meta"], sp["best_chunk"]) for sp in items]

# ---------------- Results with pagination (10 per page) ----------------
with tab_res:
    if not speeches:
        st.info("Enter a search to see results.")
    else:
        st.subheader("Most Recent" if sort_key == "newest" else "Most Relevant")

        PAGE_SIZE = 10
        total = len(speeches)
        total_pages = max(1, math.ceil(total / PAGE_SIZE))
        page = max(1, min(st.session_state.page, total_pages))

        start = (page - 1) * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)
        page_items = speeches[start:end]

        # Render page items
        for sp in page_items:
            m = sp["meta"]; ch = sp["best_chunk"]
            date_s = str(m.get('date'))
            title = m.get('title')
            link = m.get('link')
            if layout_mode == "Compact cards":
                st.markdown(f'<div class="card"><h4>{date_s} — <a href="{link}">{title}</a></h4>', unsafe_allow_html=True)
                sys = "You are an IMF speech analyst. Produce a concise snippet and 3–5 bullets grounded only in the excerpt."
                usr = f"Excerpt:\n{ch}\n\nReturn a 1–2 sentence snippet, then 3–5 bullets of key points."
                key = f"res::{date_s}::{title}::{model_preferred}"
                (md, used_model) = llm_cached(key, sys, usr, model=model_preferred,
                                              max_tokens=PER_ITEM_TOKENS, temperature=0.2,
                                              stream=True)
                st.caption(f"Model: {used_model}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"**{date_s} — [{title}]({link})**")
                sys = "You are an IMF speech analyst. Produce a concise snippet and 3–5 bullets grounded only in the excerpt."
                usr = f"Excerpt:\n{ch}\n\nReturn a 1–2 sentence snippet, then 3–5 bullets of key points."
                key = f"res::{date_s}::{title}::{model_preferred}"
                (md, used_model) = llm_cached(key, sys, usr, model=model_preferred,
                                              max_tokens=PER_ITEM_TOKENS, temperature=0.2,
                                              stream=True)
                st.caption(f"Model: {used_model}")

        # Pagination controls
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_a:
            if st.button("◀︎ Prev", disabled=(page <= 1)):
                st.session_state.page = max(1, page - 1)
                st.experimental_rerun()
        with col_b:
            st.markdown(f"<div style='text-align:center'>Page {page} of {total_pages} • Showing {start+1}–{end} of {total}</div>", unsafe_allow_html=True)
        with col_c:
            if st.button("Next ▶︎", disabled=(page >= total_pages)):
                st.session_state.page = min(total_pages, page + 1)
                st.experimental_rerun()

        with st.expander("Sources used on this page"):
            page_sources = list(sources_from_speeches(page_items))
            if not page_sources:
                st.info("No sources to show.")
            else:
                for (d, t, l) in page_sources:
                    st.markdown(f"- {d} — [{t}]({l})")
                lines = [f"{d} — {t} — {l}" for (d,t,l) in page_sources]
                all_sources_text = "\n".join(lines)
                st.markdown("**Copy all sources:**")
                st.text_area("All sources (copy)", all_sources_text, height=120, label_visibility="collapsed")
                st.download_button("Download sources (.txt)", all_sources_text.encode("utf-8"),
                                   file_name="sources.txt", mime="text/plain")

# ---------------- Thematic Evolution (year-by-year talking points + narrative) ----------------
with tab_comp:
    if not speeches:
        st.info("Enter a search to analyze thematic evolution.")
    else:
        st.subheader("Thematic Evolution (LLM) — Year-by-Year Talking Points")
        st.caption(f"Analyzing: **{filters['date_from'].isoformat()} → {filters['date_to'].isoformat()}**")

        def _within(meta, start_date, end_date):
            import datetime as dt
            try:
                d = dt.date.fromisoformat(str(meta.get("date")))
                return start_date <= d <= end_date
            except Exception:
                return False

        # Use ALL matched speeches in the chosen date range (not just current page)
        range_items = [sp for sp in speeches if _within(sp["meta"], filters['date_from'], filters['date_to'])]

        if not range_items:
            st.warning("No matching content in this date range. Try broadening the range.")
        else:
            # Group by year, preserving newest-first within each year
            def _year_of(meta):
                try:
                    return int(str(meta.get("date"))[:4])
                except Exception:
                    return None

            by_year = {}
            for sp in range_items:
                y = _year_of(sp["meta"])
                if not y:
                    continue
                by_year.setdefault(y, []).append(sp)

            # Sort years ascending and items within each year by date desc
            years_asc = sorted(by_year.keys())
            for y in years_asc:
                by_year[y] = sorted(by_year[y], key=lambda sp: _parse_date(sp["meta"].get("date")), reverse=True)

            # Build per-year context with a reasonable cap per year
            per_year_limit = 8 if len(years_asc) <= 2 else max(4, 14 // max(1, len(years_asc)))
            year_sections = []
            for y in years_asc:
                hits = _to_hits(by_year[y][:per_year_limit])
                ctx = format_hits_for_context(hits, limit=per_year_limit, char_limit=1100)
                if ctx.strip():
                    year_sections.append(f"=== Year {y} ===\n{ctx}")
            full_ctx = "\n\n".join(year_sections) if year_sections else "(no matching context)"

            # 1) Year-by-year talking points
            sys_focus = (
                "You are a senior IMF communications strategist. Using ONLY the provided context grouped by year, "
                "produce clear, issue-focused talking points for each year. Do not invent facts. "
                "When useful, cite with (Month YYYY — Title) derived from the headers in the context."
            )
            usr_focus = f"""Topic: {query}

Date range: {filters['date_from'].isoformat()} → {filters['date_to'].isoformat()}

Context grouped by year (each item starts with [YYYY-MM-DD — Title](link)):
{full_ctx}

Task:
- For each year present in the context, output a heading "### Year YYYY"
- Under each year, list 4–7 concise bullets of the **key talking points** that year, grounded strictly in the context.
- Bullets should be crisp, policy-relevant, and non-overlapping.
- If appropriate, include short cite tags like (Mar 2024 — Global Economy) using the header info.
Return clean Markdown only.
"""
            st.markdown("### Year-by-Year Talking Points")
            (focus_md, used_model1) = llm_cached(
                f"peryear_tps::{query}::{filters['date_from']}::{filters['date_to']}::{DEFAULT_MODEL}",
                sys_focus, usr_focus, model=DEFAULT_MODEL, max_tokens=PER_ITEM_TOKENS+150,
                temperature=0.2, stream=True)
            st.caption(f"Model: {used_model1}")

            # 2) Messaging evolution narrative for the whole interval
            sys_evol = (
                "Analyze how messaging evolved across the entire date range. Use ONLY the same context. "
                "Be concrete about shifts in emphasis, new themes that emerged, and any themes that receded. "
                "Cite with (Month YYYY — Title) where helpful. Avoid speculation."
            )
            usr_evol = f"""Topic: {query}
Range: {filters['date_from'].isoformat()} → {filters['date_to'].isoformat()}

Same context as above:
{full_ctx}

Task:
- Write a concise narrative (6–10 sentences) explaining how the **talking points shifted or held steady** over time.
- Be specific about what rose/fell in prominence and when.
Return a compact paragraph in Markdown.
"""
            st.markdown("### Messaging Evolution (Narrative)")
            (evol_md, used_model2) = llm_cached(
                f"evolution_narr::{query}::{filters['date_from']}::{filters['date_to']}::{DEFAULT_MODEL}",
                sys_evol, usr_evol, model=DEFAULT_MODEL, max_tokens=PER_ITEM_TOKENS,
                temperature=0.2, stream=True)
            st.caption(f"Model: {used_model2}")

            with st.expander("Show sources (Thematic Evolution)"):
                for (d, t, l) in sources_from_speeches(range_items):
                    st.markdown(f"- {d} — [{t}]({l})")

# ---------------- Optional Trajectory tab ----------------
if _HAS_P3:
    with tab_traj:
        if not speeches:
            st.info("Enter a search above to view trajectory analysis.")
        else:
            st.subheader("Messaging Trajectory")
            ctx = format_hits_for_context(_to_hits(speeches[:30]), limit=18)
            series = build_issue_trajectory(query, ctx, DEFAULT_MODEL)  # DataFrame: year, issue, share
            if not series.empty:
                fig = px.area(series, x="year", y="share", color="issue", groupnorm="fraction")
                st.plotly_chart(fig, use_container_width=True)
                forecast = forecast_issue_trends(series)
                if not forecast.empty:
                    st.markdown("**Next-year trend (simple forecast)**")
                    fig2 = px.bar(forecast, x="issue", y="forecast_share")
                    st.plotly_chart(fig2, use_container_width=True)
                nar = trajectory_narrative(query, ctx, DEFAULT_MODEL)
                st.markdown(nar)
            else:
                st.info("No trajectory could be derived from current context.")

# ---------------- Rapid Response tab ----------------
with tab_rr:
    st.subheader("Rapid Response")
    inquiry = st.text_area("Media inquiry from journalist", "")
    if st.button("Generate response") and inquiry.strip():
        try:
            rr_speeches, _ = retrieve_speeches(
                query=inquiry,
                index=index, metas=metas, chunks=chunks,
                filters=filters,
                sort="relevance",
                precision=high_precision,
                strictness=strictness,
                exact_phrase=True,
                exclude_terms=[],
                core_topic_only=core_topic_only,
                use_llm_rerank=True,
                model=DEFAULT_MODEL
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            rr_speeches = []

        rr_hits = _to_hits(rr_speeches[:30])
        ctx_rr = format_hits_for_context(rr_hits, limit=18)
        if ctx_rr.strip():
            sys = (
                "You are Kristalina Georgieva's communications aide. "
                "Using only the provided excerpts from her speeches, answer the media inquiry "
                "with 3–5 concise bullet points followed by a short narrative paragraph."
            )
            usr = f"""Inquiry: {inquiry}

Context:
{ctx_rr}

Tasks:
- Provide 3–5 bullet points addressing the inquiry.
- Then write a short narrative paragraph synthesizing the answer."""
            key = f"rr::{hashlib.sha256((inquiry+ctx_rr).encode()).hexdigest()}::{DEFAULT_MODEL}"
            (rr_md, used_model) = llm_cached(key, sys, usr, model=DEFAULT_MODEL,
                                             max_tokens=500, temperature=0.2, stream=True)
            st.caption(f"Model: {used_model}")
            with st.expander("Show sources (Rapid Response)"):
                for (d,t,l) in sources_from_speeches(rr_speeches):
                    st.markdown(f"- {d} — [{t}]({l})")
        else:
            st.warning("No relevant context found to answer the inquiry.")
    else:
        st.info("Paste the inquiry and click Generate response.")

# ---------------- Draft Assist tab ----------------
with tab_draft:
    st.subheader("First Draft Speech Assist")
    big_ideas = st.text_input("Big ideas/themes (comma-separated)", "")
    r1c1, r1c2 = st.columns(2)
    audience = r1c1.text_input("Audience", "")
    venue = r1c2.text_input("Venue", "")
    r2c1, r2c2 = st.columns(2)
    tone = r2c1.text_input("Tone", "")
    style = r2c2.text_input("Style", "")

    if st.button("Draft outline") and big_ideas.strip():
        try:
            draft_speeches, _ = retrieve_speeches(
                query=big_ideas,
                index=index, metas=metas, chunks=chunks,
                filters=filters,
                sort=sort_key,
                precision=high_precision,
                strictness=strictness,
                exact_phrase=False,
                exclude_terms=[],
                core_topic_only=core_topic_only,
                use_llm_rerank=True,
                model=DEFAULT_MODEL,
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            draft_speeches = []

        draft_hits = _to_hits(draft_speeches[:30])
        ctx = format_hits_for_context(draft_hits, limit=18)

        params = []
        if audience.strip(): params.append(f"Audience: {audience}")
        if venue.strip(): params.append(f"Venue: {venue}")
        if tone.strip(): params.append(f"Tone: {tone}")
        if style.strip(): params.append(f"Style: {style}")
        params.append(f"Big ideas: {big_ideas}")
        param_block = "\n".join(params)

        sys = "You are drafting a first-pass speech outline in the speaker's established style, grounded ONLY in provided context."
        usr = f"""Context (date-ordered excerpts):
{ctx}

{param_block}

Tasks:
- Title and 1–2 sentence setup
- Outline with sections and 2–3 bullets each (grounded in context)
- 8–10 pull quotes with (Month YYYY — Title)
- Closing paragraph
Return clean Markdown."""
        key = f"draft::{big_ideas}::{audience}::{venue}::{tone}::{style}"
        (md, used_model) = llm_cached(key, sys, usr, model=DEFAULT_MODEL,
                                      max_tokens=1200, temperature=0.3, stream=True)
        st.caption(f"Model: {used_model}")
        st.download_button("Download draft (Markdown)", md.encode("utf-8"), file_name="speech_draft.md", mime="text/markdown")
    else:
        st.info("Provide big ideas and optional parameters, then click Draft outline.")

# ---------------- Ingest & Sync tab ----------------
with tab_ing:
    st.subheader("Ingest & Sync")
    st.markdown(
        "<div class='small'>Upload new sources (PDF, DOCX, TXT/MD, HTML, audio/video) or paste a YouTube/Vimeo URL. "
        "All originals go to <code>./corpus/raw/</code>, normalized text to <code>./corpus/processed/</code>, "
        "and metadata JSON to <code>./corpus/meta/</code>. After ingest you can rebuild the index. "
        "If this folder is a Git repo with a remote, I’ll try to commit & push.</div>", unsafe_allow_html=True
    )

    # Corpus directories
    RAW_DIR = Path("corpus/raw")
    PROC_DIR = Path("corpus/processed")
    META_DIR = Path("corpus/meta")
    for p in [RAW_DIR, PROC_DIR, META_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    up_files = st.file_uploader("Upload files", type=[
        "pdf","docx","txt","md","html","htm",
        "mp3","wav","m4a","mp4","mov","avi","mkv"
    ], accept_multiple_files=True)
    url_generic = st.text_input("Optional: Generic source URL (web page, etc.)", placeholder="https://…")
    title_hint = st.text_input("Optional: Title override (if empty, filename or URL is used)")

    st.markdown("#### Fetch from YouTube/Vimeo")
    cookie_upload = st.file_uploader("YouTube cookies.txt (optional, for age/region/consent-gated videos)", type=["txt"], accept_multiple_files=False, key="yt_cookies")
    proxy_url = st.text_input("Proxy (optional)", placeholder="http://user:pass@host:port or http://host:port")
    video_url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=… or https://vimeo.com/…")
    spk_box = st.expander("Speaker filtering (optional)")
    with spk_box:
        st.caption("If the video has multiple speakers, you can filter to **only Kristalina** using diarization + voice matching.")
        enable_spk_filter = st.toggle("Filter to a specific speaker (Kristalina)", value=False)
        ref_voice = st.file_uploader("Upload a short reference audio of Kristalina (5–20s WAV/MP3 recommended)", type=["wav","mp3","m4a"])
        st.caption("Requires: pyannote.audio + a Hugging Face token in `HUGGINGFACE_TOKEN`, and speechbrain + torch.")

    colf1, colf2 = st.columns([1,2])
    with colf1:
        fetch_btn = st.button("Fetch & Transcribe")
    with colf2:
        st.caption("Uses yt-dlp to download best audio and Whisper to transcribe (needs `yt-dlp`, `ffmpeg`, and an API key or local Whisper).")

    added = []
    errors = []

    if st.button("Ingest uploaded sources"):
        # Handle file uploads
        if up_files:
            for uf in up_files:
                try:
                    slug = _safe_slug(uf.name)
                    raw_path = RAW_DIR / slug
                    raw_path.write_bytes(uf.getbuffer())

                    meta = {
                        "title": title_hint or uf.name,
                        "source": "upload",
                        "filename": uf.name,
                        "stored_at": str(raw_path),
                        "ingested_at": _now_iso(),
                        "type": Path(uf.name).suffix.lower().lstrip("."),
                    }
                    # Normalize to text
                    if raw_path.suffix.lower() in [".mp3",".wav",".m4a",".mp4",".mov",".avi",".mkv"]:
                        tr = transcribe_bytes(uf.getbuffer(), uf.name)
                        text_out = tr["text"]
                    else:
                        text_out = doc_to_text(raw_path)

                    proc_name = f"{slug}.txt"
                    proc_path = PROC_DIR / proc_name
                    proc_path.write_text(text_out, encoding="utf-8")

                    meta["normalized_text"] = str(proc_path)
                    (META_DIR / f"{slug}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    added.append(slug)
                except Exception as e:
                    errors.append(f"{uf.name}: {e}")

        # Handle simple URL capture (store URL meta; fetching left to pipeline)
        if url_generic.strip():
            try:
                slug = _safe_slug(url_generic)
                meta = {
                    "title": title_hint or url_generic,
                    "source": "url",
                    "url": url_generic,
                    "ingested_at": _now_iso(),
                    "note": "Stored URL reference; fetch/transcribe externally or paste content as .txt for now."
                }
                (META_DIR / f"{slug}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                added.append(slug)
            except Exception as e:
                errors.append(f"URL save failed: {e}")

        # Feedback
        if added:
            st.success(f"Ingested: {', '.join(added)}")
        if errors:
            st.error("Some items failed:\n" + "\n".join(errors))

        # Rebuild index if anything added
        if added:
            if build_index_main:
                with st.spinner("Rebuilding index…"):
                    try:
                        build_index_main()
                        st.success("Index rebuilt.")
                    except Exception as e:
                        st.error(f"Index rebuild failed: {e}")
            else:
                st.info("Index builder not found. Ensure build_index.py is available with `main()` entrypoint.")

            # Try git add/commit/push
            try:
                repo = git.Repo(".", search_parent_directories=True) if git else None
                if repo:
                    repo.index.add([str(RAW_DIR), str(PROC_DIR), str(META_DIR), "index.faiss", "meta.json", "chunks.json"])
                    repo.index.commit(f"Ingest via app: {len(added)} new source(s)")
                    if repo.remotes:
                        origin = repo.remotes.origin
                        origin.push()
                        st.success("Changes pushed to remote.")
                    else:
                        st.info("No git remote configured. Commit saved locally.")
                else:
                    st.info("GitPython not installed or not a git repo. Skipped commit/push.")
            except Exception as e:
                st.warning(f"Git sync skipped/failed: {e}")

        # Reload index in-session
        if added:
            try:
                index, metas, chunks = load_index()
                df = load_df()
                st.toast("Index reloaded.", icon="✅")
            except Exception as e:
                st.warning(f"Reload failed (you may need to rerun): {e}")

    # Fetch & transcribe flow
    if fetch_btn and video_url.strip():
        try:
            cookies_bytes = cookie_upload.read() if cookie_upload is not None else None
            info = ytdlp_fetch_audio(video_url, RAW_DIR, cookies_bytes=cookies_bytes, proxy=(proxy_url or None))
            raw_audio_path = info["path"]
            title = info["title"]
            # Transcribe with timestamps
            tr = transcribe_bytes(raw_audio_path.read_bytes(), raw_audio_path.name)
            text_out = tr["text"]

            # Optional: speaker filtering to Kristalina
            filtered_used = False
            if enable_spk_filter:
                if not ref_voice:
                    st.warning("Upload a reference audio to enable speaker filtering.")
                else:
                    try:
                        filtered = diarize_and_filter_audio(raw_audio_path, tr, ref_voice)
                        if filtered.get("text"):
                            text_out = filtered["text"]
                            filtered_used = True
                            st.success("Applied speaker filtering: kept segments matching the reference voice.")
                        else:
                            st.info("Speaker filtering produced no segments; using full transcript.")
                    except Exception as e:
                        st.warning(f"Speaker filtering skipped: {e}")

            proc_name = f"{_safe_slug(title)}.txt"
            proc_path = PROC_DIR / proc_name
            proc_path.write_text(text_out, encoding="utf-8")

            meta = {
                "title": title_hint or title,
                "source": "youtube_vimeo",
                "url": video_url,
                "stored_at": str(raw_audio_path),
                "normalized_text": str(proc_path),
                "uploader": info.get("uploader"),
                "upload_date": info.get("upload_date"),
                "ingested_at": _now_iso(),
                "type": "audio",
                "speaker_filtered": bool(filtered_used),
            }
            (META_DIR / f"{_safe_slug(title)}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            st.success(f"Fetched & transcribed: {title}")

            # Rebuild index
            if build_index_main:
                with st.spinner("Rebuilding index…"):
                    try:
                        build_index_main()
                        st.success("Index rebuilt.")
                    except Exception as e:
                        st.error(f"Index rebuild failed: {e}")
            else:
                st.info("Index builder not found. Ensure build_index.py exposes `main()`.")

            # Git sync
            try:
                repo = git.Repo(".", search_parent_directories=True) if git else None
                if repo:
                    repo.index.add([str(RAW_DIR), str(PROC_DIR), str(META_DIR), "index.faiss", "meta.json", "chunks.json"])
                    repo.index.commit(f"Ingest via app: fetched {title}")
                    if repo.remotes:
                        origin = repo.remotes.origin
                        origin.push()
                        st.success("Changes pushed to remote.")
                    else:
                        st.info("No git remote configured. Commit saved locally.")
                else:
                    st.info("GitPython not installed or not a git repo. Skipped commit/push.")
            except Exception as e:
                st.warning(f"Git sync skipped/failed: {e}")

            # Reload index
            try:
                index, metas, chunks = load_index()
                df = load_df()
                st.toast("Index reloaded.", icon="✅")
            except Exception as e:
                st.warning(f"Reload failed (you may need to rerun): {e}")

        except Exception as e:
            st.error(f"Fetch/transcribe failed: {e}")
            st.info("If this is a YouTube 403/age/region error, try adding a cookies.txt from your browser and/or specifying a proxy above, then retry.")
