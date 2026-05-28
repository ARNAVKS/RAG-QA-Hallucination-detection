import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdf(path: str) -> list[dict]:
    doc = fitz.open(path)
    return [{"text": page.get_text(), "page": i + 1, "source": path} for i, page in enumerate(doc)]


def chunk_text(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = [Document(page_content=p["text"], metadata={"page": p["page"], "source": p["source"]}) for p in pages]
    return splitter.split_documents(docs)
