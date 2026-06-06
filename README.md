# 🔍 RAG QA — Hallucination Detection

Question-answering over PDF documents with automatic hallucination detection.

## How It Works

```
PDF → Chunk → Embed (ChromaDB) → Retrieve → Rerank (Cohere) → LLaMA 3.3 70B (Groq) → NLI Grounding Check
```

- **Embeddings** — `all-MiniLM-L6-v2` via Sentence Transformers
- **Reranking** — Cohere Rerank filters and reorders retrieved chunks for higher relevance before generation
- **LLM** — LLaMA 3.3 70B via Groq
- **Hallucination Check** — `cross-encoder/nli-deberta-v3-base` scores each answer sentence against retrieved chunks as `ENTAILMENT`, `CONTRADICTION`, or `NEUTRAL`

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
export COHERE_API_KEY=your_key_here
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

## Hallucination Detection

Each sentence in the answer is independently checked against the retrieved context chunks using NLI:

| NLI Label | Meaning | Action |
|---|---|---|
| `ENTAILMENT` | Context directly supports the sentence | Counted toward confidence |
| `CONTRADICTION` | Context directly conflicts with the sentence | Added to `flagged_sentences` |
| `NEUTRAL` | Context neither confirms nor denies | Silently reduces confidence |

**Confidence** = `entailed sentences / total sentences`

> ⚠️ `flagged_sentences: []` does **not** mean the answer is fully grounded — neutral sentences (unverifiable claims) won't appear in flagged but will lower confidence.

## Verdict Scale

| Verdict | Confidence |
|---|---|
| `grounded` | ≥ 0.6 |
| `mostly_grounded` | 0.35 – 0.59 |
| `hallucinated` | < 0.35 |

## Stack
FastAPI · Groq · ChromaDB · Cohere · Sentence Transformers · DeBERTa · PyMuPDF · LangChain
