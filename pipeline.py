from embedder import retrieve
from llm import generate_answer
from hallucination import score_answer, HallucinationReport

_REFUSAL_PHRASE = "this information is not present in the document."


def rag_answer(question: str, doc_id: str, k: int = 5) -> str:
    chunks = retrieve(question, doc_id, k)
    return generate_answer(question, chunks)


def qa_score(question: str, doc_id: str) -> dict:
    chunks = retrieve(question, doc_id, k=20)
    answer = generate_answer(question, chunks)

    if _REFUSAL_PHRASE in answer.lower():
        return {
            "answer": answer,
            "confidence": 1.0,
            "verdict": "grounded (refusal)",
            "flagged_sentences": [],
        }

    report: HallucinationReport = score_answer(answer, chunks)
    return {
        "answer": answer,
        "confidence": report.confidence,
        "verdict": report.verdict,
        "flagged_sentences": report.flagged,
    }
