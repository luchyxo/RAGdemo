# ◈ RAG Concept Demo — Assignment 3

**Group 21** — Chove Harry (LS2525240) · Malimba Siphiwe (LS2525232) · Das Gupta Trishun (LS2525211)

Foundation Models, Hallucination, and Retrieval-Augmented Generation. Pre-loaded with Sessions 1–8 lecture content.

## Quick Start (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud (Live Link)

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo → set main file to `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_YOUR_ACTUAL_KEY"
   ```
5. Click **Deploy** — you'll get a public URL anyone can access without needing their own API key

> **Important:** Do NOT commit `.streamlit/secrets.toml` to GitHub (it's in `.gitignore`). Add the key only through the Streamlit Cloud dashboard.

## Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `session_data.json` | Pre-extracted text from 7 lecture PPTX files |
| `.streamlit/config.toml` | Dark theme and upload settings |
| `requirements.txt` | Python dependencies |

## Features

- **Session Selector** — toggle Sessions 1, 3–8 to control the knowledge base
- **3 Chunking Strategies** — fixed-size, paragraph-based, sentence-level
- **Animated Pipeline Flow** — visual document → chunks → vectors → retrieval → generation
- **Chunk Explorer** — browse every chunk with a slider
- **Retrieval Display** — ranked results with cosine similarity scores
- **RAG vs Baseline** — side-by-side comparison showing grounding reduces hallucination
- **Prompt Inspector** — view exact prompts sent to the LLM
- **Extra PDF Upload** — optionally add more documents beyond the pre-loaded sessions

## Sessions Included

| Session | Topic |
|---|---|
| 1 | The Awakening — History & Rules |
| 3 | Algorithmic DNA — Classic ML Foundations |
| 4 | Data Thinking — Quality, Distribution, Information |
| 5 | Linear Algebra & Compute Reality |
| 6 | Transformers & Foundation Models |
| 7 | Post-Training & Alignment |
| 8 | Retrieval-Augmented Generation |
