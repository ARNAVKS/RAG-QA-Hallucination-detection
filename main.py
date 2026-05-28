import os
import shutil
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from embedder import build_collection, collection_exists
from loader import chunk_text, load_pdf
from pipeline import qa_score, rag_answer

app = FastAPI(
    title="RAG QA API",
    description="Retrieval-Augmented Generation with hallucination detection",
    version="1.0.0",
)


# ---------- request/response models ----------

class AskRequest(BaseModel):
    question: str
    collection: str = "info"
    hallucination_check: bool = True


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float | None = None
    verdict: str | None = None
    flagged_sentences: list[str] | None = None


class IndexResponse(BaseModel):
    collection: str
    pages_loaded: int
    chunks_indexed: int
    message: str


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index", response_model=IndexResponse)
async def index_pdf(
    file: UploadFile = File(...),
    collection: str = Form("info"),
):
    """Upload a PDF and index it into ChromaDB. Re-uses existing collection if already indexed."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        pages = load_pdf(tmp_path)
        chunks = chunk_text(pages)
        build_collection(chunks, collection)
    finally:
        os.unlink(tmp_path)

    return IndexResponse(
        collection=collection,
        pages_loaded=len(pages),
        chunks_indexed=len(chunks),
        message=f"Collection '{collection}' is ready.",
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Ask a question against an indexed collection. Optionally runs hallucination detection."""
    if not collection_exists(req.collection):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{req.collection}' not found. Index a PDF first via POST /index.",
        )

    if req.hallucination_check:
        result = qa_score(req.question, req.collection)
        return AskResponse(
            question=req.question,
            answer=result["answer"],
            confidence=result["confidence"],
            verdict=result["verdict"],
            flagged_sentences=result["flagged_sentences"],
        )

    answer = rag_answer(req.question, req.collection)
    return AskResponse(question=req.question, answer=answer)


@app.get("/collections")
def list_collections():
    """List all indexed collections."""
    import chromadb
    client = chromadb.PersistentClient("./chromadb")
    cols = client.list_collections()
    return {"collections": [c.name for c in cols]}


