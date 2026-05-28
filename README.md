# 🔍 RAG QA — Hallucination Detection
 
Question-answering over PDF documents with automatic hallucination detection.
 
## How It Works
 
```
PDF → Chunk → Embed (ChromaDB) → Retrieve → LLaMA 3.3 70B (Groq) → NLI Grounding Check
```
 
- **Embeddings** — `all-MiniLM-L6-v2` via Sentence Transformers
- **LLM** — LLaMA 3.3 70B via Groq
- **Hallucination Check** — `cross-encoder/nli-deberta-v3-base` scores each answer sentence as `ENTAILMENT`, `CONTRADICTION`, or `NEUTRAL`
## Setup
 
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn main:app --reload
```
 
Swagger UI → `http://127.0.0.1:8000/docs`
 
## Endpoints
 
| Method | Endpoint | Description |
|---|---|---|
| POST | `/index` | Upload & index a PDF |
| POST | `/ask` | Ask a question (with optional hallucination check) |
| GET | `/collections` | List indexed collections |
| GET | `/health` | Health check |
 
## Verdict Scale
 
| Verdict | Confidence |
|---|---|
| `grounded` | ≥ 0.6 |
| `mostly_grounded` | 0.35 – 0.59 |
| `hallucinated` | < 0.35 |
 
## Stack
FastAPI · Groq · ChromaDB · Sentence Transformers · DeBERTa · PyMuPDF · LangChain
