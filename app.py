"""
Assignment 3 – RAG Concept Demo  (v2)
Foundation Models, Hallucination, and Retrieval-Augmented Generation

Pre-loaded with course lecture slides (Sessions 1–8).
Demonstrates: chunking → embedding → retrieval → generation via Groq.
"""

import streamlit as st
import numpy as np
import json, hashlib, os, re, time, math
from typing import List, Tuple
from pathlib import Path

# ---------------------------------------------------------------------------
# Session data loader
# ---------------------------------------------------------------------------

SESSION_TOPICS = {
    "Session 1": "The Awakening — History & Rules",
    "Session 3": "Algorithmic DNA — Classic ML Foundations",
    "Session 4": "Data Thinking — Quality, Distribution, Information",
    "Session 5": "Linear Algebra & Compute Reality",
    "Session 6": "Transformers & Foundation Models",
    "Session 7": "Post-Training & Alignment",
    "Session 8": "Retrieval-Augmented Generation",
}

SESSION_ICONS = {
    "Session 1": "⚡", "Session 3": "🧬", "Session 4": "📊",
    "Session 5": "🧮", "Session 6": "🤖", "Session 7": "🎯", "Session 8": "🔍",
}


@st.cache_data
def load_session_data() -> dict:
    """Load pre-extracted lecture text from the bundled JSON file."""
    data_path = Path(__file__).parent / "session_data.json"
    if data_path.exists():
        with open(data_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# PDF fallback (for additional uploads)
# ---------------------------------------------------------------------------

try:
    import fitz
    PDF_BACKEND = "pymupdf"
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PDF_BACKEND = "pypdf2"
    except ImportError:
        PDF_BACKEND = None


def extract_pdf_text(uploaded_file) -> str:
    raw = uploaded_file.read(); uploaded_file.seek(0)
    if PDF_BACKEND == "pymupdf":
        doc = fitz.open(stream=raw, filetype="pdf")
        text = "\n\n".join(p.get_text() for p in doc); doc.close(); return text
    elif PDF_BACKEND == "pypdf2":
        import io; reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_fixed(text: str, size=500, overlap=100) -> List[str]:
    chunks, s = [], 0
    while s < len(text):
        chunks.append(text[s:s+size].strip()); s += size - overlap
    return [c for c in chunks if len(c) > 20]


def chunk_paragraph(text: str, max_len=800) -> List[str]:
    paras = re.split(r"\n\s*\n", text)
    chunks, cur = [], ""
    for p in paras:
        p = p.strip()
        if not p: continue
        if len(cur) + len(p) + 1 > max_len and cur:
            chunks.append(cur.strip()); cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur.strip(): chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 20]


def chunk_sentence(text: str, n=5, overlap=1) -> List[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 5]
    chunks, i = [], 0
    while i < len(sents):
        chunks.append(" ".join(sents[i:i+n])); i += n - overlap
    return [c for c in chunks if len(c) > 20]


STRATEGIES = {
    "Fixed-Size (500 chars)": chunk_fixed,
    "Paragraph-Based": chunk_paragraph,
    "Sentence-Level (5-sent)": chunk_sentence,
}

# ---------------------------------------------------------------------------
# TF-IDF Embedder
# ---------------------------------------------------------------------------

class TFIDFEmbedder:
    def __init__(self, dim=384):
        self.dim, self.vocab, self.idf, self._proj = dim, {}, None, None

    def fit(self, corpus):
        df = {}
        for doc in corpus:
            for t in set(self._tok(doc)):
                df[t] = df.get(t, 0) + 1
                if t not in self.vocab: self.vocab[t] = len(self.vocab)
        n = len(corpus)
        self.idf = np.zeros(len(self.vocab))
        for t, idx in self.vocab.items():
            self.idf[idx] = np.log((n+1) / (df.get(t,0)+1)) + 1
        rng = np.random.RandomState(42)
        self._proj = rng.randn(len(self.vocab), self.dim).astype(np.float32)
        self._proj /= np.linalg.norm(self._proj, axis=1, keepdims=True) + 1e-9

    def embed(self, text):
        tf = np.zeros(len(self.vocab))
        toks = self._tok(text)
        for t in toks:
            if t in self.vocab: tf[self.vocab[t]] += 1
        if toks: tf /= len(toks)
        d = (tf * self.idf) @ self._proj
        n = np.linalg.norm(d)
        return d / n if n > 0 else d

    def embed_batch(self, texts):
        return np.array([self.embed(t) for t in texts])

    @staticmethod
    def _tok(text):
        return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self):
        self.vecs, self.chunks = None, []

    def add(self, chunks, vecs):
        self.chunks, self.vecs = chunks, vecs

    def search(self, qvec, k=5):
        if self.vecs is None: return []
        sims = self.vecs @ qvec
        idxs = np.argsort(sims)[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in idxs]


# ---------------------------------------------------------------------------
# Groq LLM
# ---------------------------------------------------------------------------

def call_groq(prompt, api_key, model="llama-3.1-8b-instant"):
    import urllib.request
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3, "max_tokens": 1024,
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Groq error: {e}"

RAG_PROMPT = """You are a precise technical assistant. Answer ONLY from the provided context. If the context is insufficient, say so — do not fabricate.

### Context
{context}

### Question
{question}

### Answer"""

BASELINE_PROMPT = """Answer the following question to the best of your ability.

### Question
{question}

### Answer"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_embedder(chunks):
    key = hashlib.md5("".join(chunks).encode()).hexdigest()
    if "emb_key" not in st.session_state or st.session_state.emb_key != key:
        emb = TFIDFEmbedder(); emb.fit(chunks)
        st.session_state.emb_key = key
        st.session_state.embedder = emb
    return st.session_state.embedder

def esc(t):
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br/>")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  STREAMLIT UI                                                         ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def main():
    st.set_page_config(page_title="RAG Concept Demo", page_icon="◈", layout="wide")
    inject_css()

    # ── Floating animated header ──────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="orb orb1"></div>
        <div class="orb orb2"></div>
        <div class="orb orb3"></div>
        <div class="hero-text">
            <div class="hero-tag">Assignment 3 · Group 21</div>
            <h1>RAG Concept Demo</h1>
            <p>Chunking · Embedding · Retrieval · Generation</p>
            <div class="group-members">
                <span class="member">Chove Harry <em>LS2525240</em></span>
                <span class="member-dot">·</span>
                <span class="member">Malimba Siphiwe <em>LS2525232</em></span>
                <span class="member-dot">·</span>
                <span class="member">Das Gupta Trishun <em>LS2525211</em></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    # API key: try Streamlit secrets first, fall back to manual input
    _secret_key = ""
    try:
        _secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass

    with st.sidebar:
        st.markdown('<div class="sidebar-title">◈ Controls</div>', unsafe_allow_html=True)
        if _secret_key:
            api_key = _secret_key
            st.success("API key loaded ✓", icon="🔑")
        else:
            api_key = st.text_input("Groq API Key", type="password",
                                    help="Free at console.groq.com")
        model = st.selectbox("Model", [
            "llama-3.1-8b-instant", "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768", "gemma2-9b-it",
        ])
        strategy = st.selectbox("Chunking Strategy", list(STRATEGIES.keys()))
        top_k = st.slider("Retrieve top-K", 1, 10, 4)

    # ── Session selector ──────────────────────────────────────────────────
    sessions = load_session_data()

    if not sessions:
        st.error("session_data.json not found next to app.py. "
                 "Place the file in the same directory and reload.")
        return

    st.markdown('<div class="section-label">Select Knowledge Sources</div>',
                unsafe_allow_html=True)

    # Build the pill-selector HTML
    if "selected_sessions" not in st.session_state:
        st.session_state.selected_sessions = list(sessions.keys())

    cols = st.columns(len(sessions))
    for i, (skey, _) in enumerate(sessions.items()):
        with cols[i]:
            icon = SESSION_ICONS.get(skey, "📄")
            label = skey.replace("Session ", "S")
            checked = st.checkbox(f"{icon} {label}", value=True, key=f"chk_{skey}")

    active = [k for k in sessions if st.session_state.get(f"chk_{k}", True)]

    if not active:
        st.warning("Select at least one session.")
        return

    # ── Combine selected text ─────────────────────────────────────────────
    combined = "\n\n".join(sessions[k] for k in active)

    # Optional extra PDFs
    extra = st.file_uploader("➕ Upload additional PDFs (optional)", type=["pdf"],
                             accept_multiple_files=True)
    if extra:
        for f in extra:
            combined += "\n\n" + extract_pdf_text(f)

    # ── Chunk & Embed ─────────────────────────────────────────────────────
    chunks = STRATEGIES[strategy](combined)
    embedder = build_embedder(chunks)
    vecs = embedder.embed_batch(chunks)
    store = VectorStore()
    store.add(chunks, vecs)

    # ── Animated pipeline flow ────────────────────────────────────────────
    st.markdown('<div class="section-label">Pipeline Overview</div>',
                unsafe_allow_html=True)

    render_pipeline_flow(len(active), len(combined), len(chunks), vecs.shape[1])

    # ── Chunk explorer ────────────────────────────────────────────────────
    with st.expander("🧩 Chunk Explorer"):
        idx = st.slider("Browse chunks", 0, len(chunks)-1, 0)
        st.markdown(f'<div class="chunk-bubble">{esc(chunks[idx])}</div>',
                    unsafe_allow_html=True)
        st.caption(f"Chunk {idx+1}/{len(chunks)}  ·  {len(chunks[idx])} chars")

    # ── Query ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)
    question = st.text_input("", placeholder="e.g.  What is the KV cache and why does it matter?",
                             label_visibility="collapsed")

    if not question:
        st.markdown('<p class="hint">Type a question to trigger retrieval & generation.</p>',
                    unsafe_allow_html=True)
        return
    if not api_key:
        st.warning("Enter your Groq API key in the sidebar.")
        return

    # ── Retrieval ─────────────────────────────────────────────────────────
    qvec = embedder.embed(question)
    results = store.search(qvec, top_k)

    st.markdown('<div class="section-label">Retrieved Chunks</div>', unsafe_allow_html=True)
    render_retrieval_bubbles(results)

    # ── Generation ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Generation — RAG vs Baseline</div>',
                unsafe_allow_html=True)

    ctx = "\n---\n".join(c for c, _ in results)
    rag_p = RAG_PROMPT.format(context=ctx, question=question)
    base_p = BASELINE_PROMPT.format(question=question)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="gen-label rag-label">● RAG-Augmented</div>',
                    unsafe_allow_html=True)
        with st.spinner("Generating with context…"):
            rag_ans = call_groq(rag_p, api_key, model)
        st.markdown(f'<div class="answer-blob rag-blob">{esc(rag_ans)}</div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="gen-label base-label">● Baseline (no retrieval)</div>',
                    unsafe_allow_html=True)
        with st.spinner("Generating without context…"):
            base_ans = call_groq(base_p, api_key, model)
        st.markdown(f'<div class="answer-blob base-blob">{esc(base_ans)}</div>',
                    unsafe_allow_html=True)

    # ── Prompt inspector ──────────────────────────────────────────────────
    with st.expander("🔎 Prompt Inspector"):
        t1, t2 = st.tabs(["RAG Prompt", "Baseline Prompt"])
        with t1: st.code(rag_p, language="text")
        with t2: st.code(base_p, language="text")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  UI COMPONENTS                                                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def render_pipeline_flow(n_sessions, n_chars, n_chunks, emb_dim):
    """Animated hexagonal pipeline flow."""
    st.markdown(f"""
    <div class="flow-row">
        <div class="hex-node">
            <div class="hex-icon">📄</div>
            <div class="hex-val">{n_sessions}</div>
            <div class="hex-lbl">Sessions</div>
        </div>
        <div class="flow-arrow">
            <svg viewBox="0 0 60 20"><path d="M0 10 L45 10 M38 4 L48 10 L38 16" stroke="var(--accent)" stroke-width="2" fill="none"/></svg>
        </div>
        <div class="hex-node">
            <div class="hex-icon">✂️</div>
            <div class="hex-val">{n_chunks}</div>
            <div class="hex-lbl">Chunks</div>
        </div>
        <div class="flow-arrow">
            <svg viewBox="0 0 60 20"><path d="M0 10 L45 10 M38 4 L48 10 L38 16" stroke="var(--accent)" stroke-width="2" fill="none"/></svg>
        </div>
        <div class="hex-node">
            <div class="hex-icon">🧬</div>
            <div class="hex-val">{emb_dim}d</div>
            <div class="hex-lbl">Vectors</div>
        </div>
        <div class="flow-arrow">
            <svg viewBox="0 0 60 20"><path d="M0 10 L45 10 M38 4 L48 10 L38 16" stroke="var(--accent)" stroke-width="2" fill="none"/></svg>
        </div>
        <div class="hex-node">
            <div class="hex-icon">🎯</div>
            <div class="hex-val">cosine</div>
            <div class="hex-lbl">Retrieval</div>
        </div>
        <div class="flow-arrow">
            <svg viewBox="0 0 60 20"><path d="M0 10 L45 10 M38 4 L48 10 L38 16" stroke="var(--accent)" stroke-width="2" fill="none"/></svg>
        </div>
        <div class="hex-node">
            <div class="hex-icon">🤖</div>
            <div class="hex-val">LLM</div>
            <div class="hex-lbl">Generate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_retrieval_bubbles(results):
    """Render retrieved chunks as floating score-sized bubbles."""
    html = '<div class="retrieval-grid">'
    for rank, (chunk, score) in enumerate(results, 1):
        hue = 160 + int((1 - score) * 60)  # green → teal
        bg = f"linear-gradient(135deg, hsl({hue} 55% 14%) 0%, hsl({hue} 40% 10%) 100%)"
        bd = f"1px solid hsl({hue} 50% 25% / 0.5)"
        html += f"""
        <div class="ret-bubble" style="background:{bg};border:{bd};animation-delay:{rank*0.08}s">
            <div class="ret-rank">#{rank}</div>
            <div class="ret-score">{score:.3f}</div>
            <div class="ret-text">{esc(chunk[:300])}{'…' if len(chunk)>300 else ''}</div>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  CSS                                                                  ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --bg: #080c14;
        --surface: #0e1420;
        --border: #1a2535;
        --text: #cdd6e4;
        --muted: #5a6a7f;
        --accent: #00e5a0;
        --accent2: #00b4d8;
        --warn: #ff6b35;
    }

    .block-container { max-width: 1120px; }
    h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; }

    /* ── Hero ───────────────────────────────────────────────── */
    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 24px;
        padding: 3rem 2.5rem 2.5rem;
        background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #0a1628 100%);
        margin-bottom: 2rem;
        border: 1px solid var(--border);
    }
    .hero-text { position: relative; z-index: 2; }
    .hero-tag {
        display: inline-block;
        background: var(--accent);
        color: #080c14;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 20px;
        margin-bottom: 0.8rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem;
    }
    .hero p { color: var(--muted); font-size: 1rem; margin: 0; }
    .group-members {
        display: flex;
        align-items: center;
        gap: 0;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }
    .member {
        font-size: 0.82rem;
        color: #8899aa;
    }
    .member em {
        font-style: normal;
        color: var(--accent2);
        font-family: 'DM Mono', monospace;
        font-size: 0.74rem;
        margin-left: 4px;
    }
    .member-dot {
        color: #3a4a5f;
        margin: 0 10px;
        font-size: 0.9rem;
    }

    /* floating orbs */
    .orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(60px);
        opacity: 0.35;
        animation: drift 8s ease-in-out infinite alternate;
    }
    .orb1 { width:220px;height:220px;background:var(--accent);top:-40px;right:10%;animation-delay:0s; }
    .orb2 { width:180px;height:180px;background:var(--accent2);bottom:-30px;left:15%;animation-delay:2s; }
    .orb3 { width:120px;height:120px;background:#7c3aed;top:30%;left:50%;animation-delay:4s; }
    @keyframes drift {
        0%   { transform: translate(0,0) scale(1); }
        100% { transform: translate(30px,20px) scale(1.15); }
    }

    /* ── Section labels ─────────────────────────────────────── */
    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        color: var(--accent);
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin: 2rem 0 1rem;
        padding-left: 14px;
        border-left: 3px solid var(--accent);
    }

    /* ── Pipeline flow row ──────────────────────────────────── */
    .flow-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        flex-wrap: wrap;
        margin: 0.5rem 0 1.5rem;
    }
    .hex-node {
        text-align: center;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.1rem 0.8rem;
        min-width: 90px;
        animation: nodeIn 0.6s ease both;
    }
    .hex-node:nth-child(1)  { animation-delay: 0.0s; }
    .hex-node:nth-child(3)  { animation-delay: 0.12s; }
    .hex-node:nth-child(5)  { animation-delay: 0.24s; }
    .hex-node:nth-child(7)  { animation-delay: 0.36s; }
    .hex-node:nth-child(9)  { animation-delay: 0.48s; }
    @keyframes nodeIn {
        from { opacity:0; transform: translateY(16px) scale(0.9); }
        to   { opacity:1; transform: translateY(0) scale(1); }
    }
    .hex-icon { font-size: 1.4rem; }
    .hex-val  { font-family:'DM Mono',monospace; font-size:1.15rem; font-weight:700; color:var(--accent); }
    .hex-lbl  { font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; }
    .flow-arrow { width: 48px; display: flex; align-items: center; justify-content: center; }
    .flow-arrow svg { width: 48px; height: 20px; }

    /* ── Chunk bubble ───────────────────────────────────────── */
    .chunk-bubble {
        background: linear-gradient(145deg, #0e1a2a 0%, #111e30 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        line-height: 1.6;
        color: var(--text);
        max-height: 220px;
        overflow-y: auto;
    }

    /* ── Retrieval bubbles ──────────────────────────────────── */
    .retrieval-grid {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .ret-bubble {
        border-radius: 20px;
        padding: 0.9rem 1.3rem;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        animation: bubbleIn 0.4s ease both;
    }
    @keyframes bubbleIn {
        from { opacity:0; transform: translateX(-20px); }
        to   { opacity:1; transform: translateX(0); }
    }
    .ret-rank {
        background: var(--accent);
        color: var(--bg);
        font-weight: 700;
        font-size: 0.72rem;
        padding: 3px 10px;
        border-radius: 12px;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .ret-score {
        font-family: 'DM Mono', monospace;
        color: var(--accent2);
        font-size: 0.78rem;
        flex-shrink: 0;
        min-width: 50px;
        margin-top: 2px;
    }
    .ret-text {
        font-size: 0.82rem;
        color: var(--text);
        line-height: 1.55;
    }

    /* ── Gen labels & blobs ─────────────────────────────────── */
    .gen-label { font-weight:700; font-size:0.9rem; margin-bottom:0.5rem; }
    .rag-label { color: var(--accent); }
    .base-label { color: var(--warn); }

    .answer-blob {
        border-radius: 20px;
        padding: 1.2rem 1.4rem;
        font-size: 0.88rem;
        line-height: 1.65;
        color: var(--text);
        min-height: 100px;
    }
    .rag-blob {
        background: linear-gradient(145deg, #091f16 0%, #0e2a1e 100%);
        border: 1px solid #1a5c3a;
    }
    .base-blob {
        background: linear-gradient(145deg, #1f1208 0%, #2a1a0d 100%);
        border: 1px solid #7a3d12;
    }

    /* ── Sidebar ────────────────────────────────────────────── */
    .sidebar-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--accent);
        margin-bottom: 1rem;
    }

    .hint { color: var(--muted); font-style: italic; font-size: 0.88rem; }

    /* ── Responsiveness ─────────────────────────────────────── */
    @media (max-width: 700px) {
        .flow-row { gap: 6px; }
        .hex-node { min-width: 64px; padding: 0.6rem 0.5rem; }
        .flow-arrow { width: 24px; }
        .hero h1 { font-size: 1.6rem; }
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
