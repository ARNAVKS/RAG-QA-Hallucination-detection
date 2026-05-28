import os
from groq import Groq

_SYSTEM = """You are a document assistant.
Answer ONLY using the context provided below.
If the answer is not in the context, say exactly:
'This information is not present in the document.'
Do not use outside knowledge under any circumstances."""

_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))


def generate_answer(question: str, chunks: list[str], model: str = "llama-3.3-70b-versatile") -> str:
    context = "\n\n---\n\n".join(chunks)
    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content
