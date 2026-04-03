"""
Assignment 3 – RAG Concept Demo  (v3 – UI Overhaul)
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
 
    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-grid"></div>
        <div class="hero-glow hero-glow-1"></div>
        <div class="hero-glow hero-glow-2"></div>
        <div class="hero-glow hero-glow-3"></div>
        <div class="hero-content">
            <div class="hero-badge">
                <span class="badge-dot"></span>
                Assignment 3 &middot; Group 21
            </div>
            <h1 class="hero-title">RAG Concept<br/><span class="hero-title-alt">Demo</span></h1>
            <p class="hero-subtitle">Retrieval-Augmented Generation pipeline — from chunking to generation,<br/>visualised end to end.</p>
            <div class="hero-members">
                <div class="member-chip">
                    <span class="member-avatar">HC</span>
                    <div class="member-info"><span class="member-name">Chove Harry</span><span class="member-id">LS2525240</span></div>
                </div>
                <div class="member-chip">
                    <span class="member-avatar">MS</span>
                    <div class="member-info"><span class="member-name">Malimba Siphiwe</span><span class="member-id">LS2525232</span></div>
                </div>
                <div class="member-chip">
                    <span class="member-avatar">DG</span>
                    <div class="member-info"><span class="member-name">Das Gupta Trishun</span><span class="member-id">LS2525211</span></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Sidebar ───────────────────────────────────────────────────────────
    _secret_key = ""
    try:
        _secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
 
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-logo">◈</div>
            <div class="sidebar-brand">Pipeline Controls</div>
        </div>
        """, unsafe_allow_html=True)
 
        if _secret_key:
            api_key = _secret_key
            st.markdown('<div class="api-status api-ok"><span class="status-dot status-ok"></span>API key active</div>', unsafe_allow_html=True)
        else:
            api_key = st.text_input("Groq API Key", type="password",
                                    help="Free at console.groq.com")
 
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="control-label">Model</div>', unsafe_allow_html=True)
        model = st.selectbox("Model", [
            "llama-3.1-8b-instant", "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768", "gemma2-9b-it",
        ], label_visibility="collapsed")
 
        st.markdown('<div class="control-label">Chunking Strategy</div>', unsafe_allow_html=True)
        strategy = st.selectbox("Chunking Strategy", list(STRATEGIES.keys()), label_visibility="collapsed")
 
        st.markdown('<div class="control-label">Retrieve top-K</div>', unsafe_allow_html=True)
        top_k = st.slider("Retrieve top-K", 1, 10, 4, label_visibility="collapsed")
 
    # ── Session selector ──────────────────────────────────────────────────
    sessions = load_session_data()
 
    if not sessions:
        st.error("session_data.json not found next to app.py. "
                 "Place the file in the same directory and reload.")
        return
 
    st.markdown('<div class="section-head"><span class="section-num">01</span><span class="section-title">Knowledge Sources</span></div>', unsafe_allow_html=True)
 
    if "selected_sessions" not in st.session_state:
        st.session_state.selected_sessions = list(sessions.keys())
 
    cols = st.columns(len(sessions))
    for i, (skey, _) in enumerate(sessions.items()):
        with cols[i]:
            icon = SESSION_ICONS.get(skey, "📄")
            label = skey.replace("Session ", "S")
            topic = SESSION_TOPICS.get(skey, "").split("—")[-1].strip() if skey in SESSION_TOPICS else ""
            checked = st.checkbox(f"{icon} {label}", value=True, key=f"chk_{skey}")
 
    active = [k for k in sessions if st.session_state.get(f"chk_{k}", True)]
 
    if not active:
        st.warning("Select at least one session.")
        return
 
    # ── Combine selected text ─────────────────────────────────────────────
    combined = "\n\n".join(sessions[k] for k in active)
 
    extra = st.file_uploader("Upload additional PDFs (optional)", type=["pdf"],
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
 
    # ── Pipeline visualisation ────────────────────────────────────────────
    st.markdown('<div class="section-head"><span class="section-num">02</span><span class="section-title">Pipeline</span></div>', unsafe_allow_html=True)
 
    render_pipeline(len(active), len(combined), len(chunks), vecs.shape[1])
 
    # ── Stats ribbon ──────────────────────────────────────────────────────
    render_stats_ribbon(len(active), len(combined), len(chunks), vecs.shape[1])
 
    # ── Chunk explorer ────────────────────────────────────────────────────
    st.markdown('<div class="section-head"><span class="section-num">03</span><span class="section-title">Chunk Explorer</span></div>', unsafe_allow_html=True)
 
    idx = st.slider("Browse chunks", 0, len(chunks)-1, 0, label_visibility="collapsed")
    st.markdown(f"""
    <div class="chunk-card">
        <div class="chunk-header">
            <span class="chunk-tag">Chunk {idx+1} of {len(chunks)}</span>
            <span class="chunk-meta">{len(chunks[idx])} chars</span>
        </div>
        <div class="chunk-body">{esc(chunks[idx])}</div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Query ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-head"><span class="section-num">04</span><span class="section-title">Query</span></div>', unsafe_allow_html=True)
 
    question = st.text_input("", placeholder="e.g.  What is the KV cache and why does it matter?",
                             label_visibility="collapsed")
 
    if not question:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">⌨</div>
            <div class="empty-msg">Type a question above to trigger retrieval &amp; generation</div>
        </div>
        """, unsafe_allow_html=True)
        return
    if not api_key:
        st.warning("Enter your Groq API key in the sidebar.")
        return
 
    # ── Retrieval ─────────────────────────────────────────────────────────
    qvec = embedder.embed(question)
    results = store.search(qvec, top_k)
 
    st.markdown('<div class="section-head"><span class="section-num">05</span><span class="section-title">Retrieved Chunks</span></div>', unsafe_allow_html=True)
    render_retrieval_cards(results)
 
    # ── Generation ────────────────────────────────────────────────────────
    st.markdown('<div class="section-head"><span class="section-num">06</span><span class="section-title">Generation — RAG vs Baseline</span></div>', unsafe_allow_html=True)
 
    ctx = "\n---\n".join(c for c, _ in results)
    rag_p = RAG_PROMPT.format(context=ctx, question=question)
    base_p = BASELINE_PROMPT.format(question=question)
 
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="gen-badge gen-rag">
            <span class="gen-dot gen-dot-rag"></span> RAG-Augmented
        </div>""", unsafe_allow_html=True)
        with st.spinner("Generating with context…"):
            rag_ans = call_groq(rag_p, api_key, model)
        st.markdown(f'<div class="answer-card answer-rag">{esc(rag_ans)}</div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="gen-badge gen-base">
            <span class="gen-dot gen-dot-base"></span> Baseline (no retrieval)
        </div>""", unsafe_allow_html=True)
        with st.spinner("Generating without context…"):
            base_ans = call_groq(base_p, api_key, model)
        st.markdown(f'<div class="answer-card answer-base">{esc(base_ans)}</div>',
                    unsafe_allow_html=True)
 
    # ── Prompt inspector ──────────────────────────────────────────────────
    with st.expander("🔎 Prompt Inspector"):
        t1, t2 = st.tabs(["RAG Prompt", "Baseline Prompt"])
        with t1: st.code(rag_p, language="text")
        with t2: st.code(base_p, language="text")
 
 
# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  UI COMPONENTS                                                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝
 
def render_pipeline(n_sessions, n_chars, n_chunks, emb_dim):
    """Animated glowing pipeline with connection beams."""
    nodes = [
        ("📄", str(n_sessions), "Sources"),
        ("✂️", str(n_chunks), "Chunks"),
        ("🧬", f"{emb_dim}d", "Vectors"),
        ("🎯", "cosine", "Retrieval"),
        ("🤖", "LLM", "Generate"),
    ]
    html = '<div class="pipeline">'
    for i, (icon, val, lbl) in enumerate(nodes):
        if i > 0:
            html += f'<div class="pipe-conn"><div class="pipe-beam" style="animation-delay:{i*0.15}s"></div></div>'
        html += f"""
        <div class="pipe-node" style="animation-delay:{i*0.1}s">
            <div class="pipe-ring"></div>
            <div class="pipe-icon">{icon}</div>
            <div class="pipe-val">{val}</div>
            <div class="pipe-lbl">{lbl}</div>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
 
 
def render_stats_ribbon(n_sess, n_chars, n_chunks, dim):
    """Compact stats bar below pipeline."""
    st.markdown(f"""
    <div class="stats-ribbon">
        <div class="stat"><span class="stat-val">{n_sess}</span><span class="stat-lbl">sessions</span></div>
        <div class="stat-sep"></div>
        <div class="stat"><span class="stat-val">{n_chars:,}</span><span class="stat-lbl">characters</span></div>
        <div class="stat-sep"></div>
        <div class="stat"><span class="stat-val">{n_chunks}</span><span class="stat-lbl">chunks</span></div>
        <div class="stat-sep"></div>
        <div class="stat"><span class="stat-val">{dim}</span><span class="stat-lbl">dimensions</span></div>
    </div>
    """, unsafe_allow_html=True)
 
 
def render_retrieval_cards(results):
    """Render retrieved chunks as glass cards with score bars."""
    html = '<div class="ret-list">'
    max_score = max(s for _, s in results) if results else 1
    for rank, (chunk, score) in enumerate(results, 1):
        pct = (score / max_score) * 100 if max_score > 0 else 0
        html += f"""
        <div class="ret-card" style="animation-delay:{rank*0.07}s">
            <div class="ret-left">
                <div class="ret-rank-badge">#{rank}</div>
                <div class="ret-score-wrap">
                    <div class="ret-score-bar" style="width:{pct}%"></div>
                </div>
                <div class="ret-score-val">{score:.4f}</div>
            </div>
            <div class="ret-right">
                <div class="ret-text">{esc(chunk[:350])}{'…' if len(chunk)>350 else ''}</div>
            </div>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
 
 
# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  CSS                                                                  ║
# ╚═════════════════════════════════════════════════════════════════════════╝
 
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600&display=swap');
 
    :root {
        --bg-deep:    #050709;
        --bg:         #0a0d12;
        --surface:    #0f1319;
        --surface-2:  #141a22;
        --border:     #1c2433;
        --border-lit: #2a3545;
        --text:       #d0d8e4;
        --text-dim:   #6b7a8d;
        --text-xdim:  #3d4d5f;
        --mint:       #34d399;
        --mint-dim:   #1a6b4d;
        --cyan:       #22d3ee;
        --cyan-dim:   #0e6b77;
        --violet:     #a78bfa;
        --amber:      #fbbf24;
        --rose:       #fb7185;
    }
 
    /* ── Global resets ──────────────────────────────────────── */
    .block-container { max-width: 1100px; padding-top: 2rem; }
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg-deep) !important;
    }
    [data-testid="stSidebar"] {
        background: var(--bg) !important;
        border-right: 1px solid var(--border) !important;
    }
    h1,h2,h3,h4,h5,h6 { font-family: 'Syne', sans-serif !important; }
    p, li, div { font-family: 'Outfit', sans-serif; }
 
    /* ── Hero ───────────────────────────────────────────────── */
    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 20px;
        padding: 3.5rem 3rem 3rem;
        background: var(--surface);
        border: 1px solid var(--border);
        margin-bottom: 2.5rem;
    }
    .hero-grid {
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(var(--border) 1px, transparent 1px),
            linear-gradient(90deg, var(--border) 1px, transparent 1px);
        background-size: 40px 40px;
        opacity: 0.25;
        mask-image: radial-gradient(ellipse 70% 60% at 50% 40%, black 30%, transparent 100%);
        -webkit-mask-image: radial-gradient(ellipse 70% 60% at 50% 40%, black 30%, transparent 100%);
    }
    .hero-glow {
        position: absolute;
        border-radius: 50%;
        filter: blur(80px);
        opacity: 0.18;
        animation: glowDrift 10s ease-in-out infinite alternate;
    }
    .hero-glow-1 { width:300px;height:300px;background:var(--mint);top:-80px;right:5%; }
    .hero-glow-2 { width:250px;height:250px;background:var(--cyan);bottom:-60px;left:10%;animation-delay:3s; }
    .hero-glow-3 { width:160px;height:160px;background:var(--violet);top:20%;left:55%;animation-delay:6s; }
    @keyframes glowDrift {
        0%   { transform: translate(0,0) scale(1); }
        100% { transform: translate(20px,15px) scale(1.12); }
    }
    .hero-content { position: relative; z-index: 2; }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(52,211,153,0.1);
        border: 1px solid rgba(52,211,153,0.25);
        color: var(--mint);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 5px 16px;
        border-radius: 99px;
        margin-bottom: 1.2rem;
    }
    .badge-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--mint);
        animation: blink 2s ease-in-out infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    .hero-title {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800;
        font-size: 3rem;
        line-height: 1.1;
        color: #f0f4f8;
        margin: 0 0 0.6rem;
        letter-spacing: -0.02em;
    }
    .hero-title-alt {
        background: linear-gradient(135deg, var(--mint), var(--cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: var(--text-dim);
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0 0 1.5rem;
        max-width: 540px;
    }
    .hero-members {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .member-chip {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 6px 14px 6px 6px;
        transition: border-color 0.3s;
    }
    .member-chip:hover { border-color: var(--border-lit); }
    .member-avatar {
        width: 32px; height: 32px;
        border-radius: 8px;
        background: linear-gradient(135deg, var(--mint-dim), var(--cyan-dim));
        color: #fff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        letter-spacing: 0.04em;
    }
    .member-info { display: flex; flex-direction: column; }
    .member-name {
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text);
        line-height: 1.2;
    }
    .member-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: var(--text-dim);
    }
 
    /* ── Sidebar ────────────────────────────────────────────── */
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border);
    }
    .sidebar-logo {
        width: 36px; height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--mint), var(--cyan));
        color: var(--bg-deep);
        font-size: 1.1rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .sidebar-brand {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        color: var(--text);
    }
    .sidebar-divider {
        height: 1px;
        background: var(--border);
        margin: 1rem 0;
    }
    .control-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
        margin-top: 0.6rem;
    }
    .api-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        padding: 8px 14px;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .api-ok {
        background: rgba(52,211,153,0.08);
        border: 1px solid rgba(52,211,153,0.2);
        color: var(--mint);
    }
    .status-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .status-ok { background: var(--mint); box-shadow: 0 0 8px var(--mint); }
 
    /* ── Section heads ──────────────────────────────────────── */
    .section-head {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 2.5rem 0 1.2rem;
    }
    .section-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--mint);
        background: rgba(52,211,153,0.1);
        border: 1px solid rgba(52,211,153,0.2);
        padding: 3px 10px;
        border-radius: 6px;
        letter-spacing: 0.05em;
    }
    .section-title {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        color: #e8edf4;
        letter-spacing: -0.01em;
    }
 
    /* ── Pipeline ───────────────────────────────────────────── */
    .pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        flex-wrap: wrap;
        padding: 1.5rem 0;
    }
    .pipe-node {
        position: relative;
        text-align: center;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem 1.3rem 1rem;
        min-width: 100px;
        animation: nodeUp 0.5s ease both;
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    .pipe-node:hover {
        border-color: var(--mint);
        box-shadow: 0 0 20px rgba(52,211,153,0.1);
    }
    @keyframes nodeUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .pipe-ring {
        position: absolute;
        inset: -1px;
        border-radius: 16px;
        background: conic-gradient(from 0deg, transparent, var(--mint), transparent, var(--cyan), transparent);
        opacity: 0;
        transition: opacity 0.3s;
        z-index: -1;
    }
    .pipe-node:hover .pipe-ring { opacity: 0.3; }
    .pipe-icon { font-size: 1.5rem; margin-bottom: 4px; }
    .pipe-val  {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--mint);
    }
    .pipe-lbl  {
        font-family: 'Outfit', sans-serif;
        font-size: 0.68rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 2px;
    }
    .pipe-conn {
        width: 48px;
        height: 3px;
        background: var(--border);
        position: relative;
        border-radius: 2px;
        overflow: hidden;
    }
    .pipe-beam {
        position: absolute;
        top: 0; left: -40%;
        width: 40%;
        height: 100%;
        background: linear-gradient(90deg, transparent, var(--mint), transparent);
        animation: beamSlide 2s ease-in-out infinite;
        border-radius: 2px;
    }
    @keyframes beamSlide {
        0%   { left: -40%; }
        100% { left: 100%; }
    }
 
    /* ── Stats ribbon ───────────────────────────────────────── */
    .stats-ribbon {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.8rem 2rem;
        margin-bottom: 1rem;
    }
    .stat {
        display: flex;
        align-items: baseline;
        gap: 6px;
        padding: 0 1.2rem;
    }
    .stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: var(--cyan);
    }
    .stat-lbl {
        font-family: 'Outfit', sans-serif;
        font-size: 0.72rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .stat-sep {
        width: 1px;
        height: 20px;
        background: var(--border);
    }
 
    /* ── Chunk card ─────────────────────────────────────────── */
    .chunk-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        overflow: hidden;
    }
    .chunk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.7rem 1.2rem;
        background: var(--surface-2);
        border-bottom: 1px solid var(--border);
    }
    .chunk-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--mint);
    }
    .chunk-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: var(--text-dim);
    }
    .chunk-body {
        padding: 1rem 1.2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        line-height: 1.7;
        color: var(--text);
        max-height: 200px;
        overflow-y: auto;
    }
 
    /* ── Empty state ────────────────────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        opacity: 0.5;
    }
    .empty-icon { font-size: 2rem; margin-bottom: 0.6rem; }
    .empty-msg {
        font-family: 'Outfit', sans-serif;
        font-size: 0.88rem;
        color: var(--text-dim);
    }
 
    /* ── Retrieval cards ────────────────────────────────────── */
    .ret-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .ret-card {
        display: flex;
        align-items: stretch;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
        animation: cardIn 0.35s ease both;
        transition: border-color 0.3s;
    }
    .ret-card:hover { border-color: var(--border-lit); }
    @keyframes cardIn {
        from { opacity: 0; transform: translateX(-12px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    .ret-left {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 6px;
        padding: 0.8rem 1rem;
        background: var(--surface-2);
        min-width: 80px;
        border-right: 1px solid var(--border);
    }
    .ret-rank-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--bg-deep);
        background: var(--mint);
        padding: 2px 10px;
        border-radius: 8px;
    }
    .ret-score-wrap {
        width: 48px;
        height: 4px;
        background: var(--border);
        border-radius: 2px;
        overflow: hidden;
    }
    .ret-score-bar {
        height: 100%;
        background: linear-gradient(90deg, var(--mint), var(--cyan));
        border-radius: 2px;
        transition: width 0.6s ease;
    }
    .ret-score-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: var(--cyan);
    }
    .ret-right {
        padding: 0.8rem 1.2rem;
        flex: 1;
    }
    .ret-text {
        font-family: 'Outfit', sans-serif;
        font-size: 0.82rem;
        color: var(--text);
        line-height: 1.6;
    }
 
    /* ── Generation cards ───────────────────────────────────── */
    .gen-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 8px;
        margin-bottom: 0.7rem;
    }
    .gen-rag {
        color: var(--mint);
        background: rgba(52,211,153,0.08);
        border: 1px solid rgba(52,211,153,0.2);
    }
    .gen-base {
        color: var(--amber);
        background: rgba(251,191,36,0.08);
        border: 1px solid rgba(251,191,36,0.2);
    }
    .gen-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
    }
    .gen-dot-rag  { background: var(--mint); box-shadow: 0 0 6px var(--mint); }
    .gen-dot-base { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
 
    .answer-card {
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        font-family: 'Outfit', sans-serif;
        font-size: 0.88rem;
        line-height: 1.7;
        color: var(--text);
        min-height: 120px;
    }
    .answer-rag {
        background: linear-gradient(145deg, rgba(52,211,153,0.04) 0%, rgba(34,211,238,0.03) 100%);
        border: 1px solid rgba(52,211,153,0.15);
    }
    .answer-base {
        background: linear-gradient(145deg, rgba(251,191,36,0.04) 0%, rgba(251,113,133,0.03) 100%);
        border: 1px solid rgba(251,191,36,0.15);
    }
 
    /* ── Scrollbar ──────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-lit); border-radius: 3px; }
 
    /* ── Responsiveness ─────────────────────────────────────── */
    @media (max-width: 700px) {
        .hero { padding: 2rem 1.5rem; }
        .hero-title { font-size: 2rem; }
        .hero-members { flex-direction: column; }
        .pipeline { gap: 4px; }
        .pipe-node { min-width: 70px; padding: 0.8rem 0.6rem; }
        .pipe-conn { width: 24px; }
        .stats-ribbon { flex-wrap: wrap; gap: 8px; padding: 0.6rem 1rem; }
        .stat { padding: 0 0.6rem; }
    }
    </style>
    """, unsafe_allow_html=True)
 
 
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    
