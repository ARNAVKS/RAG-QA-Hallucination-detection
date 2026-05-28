import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_client = chromadb.PersistentClient("./chromadb")


def build_collection(chunks: list[Document], doc_id: str) -> chromadb.Collection:
    lib = _client.get_or_create_collection(doc_id)
    if lib.count() > 0:
        print(f"Collection '{doc_id}' already exists with {lib.count()} entries.")
        return lib

    texts = [c.page_content for c in chunks]
    embeddings = embed_model.encode(texts).tolist()
    metadatas = [c.metadata for c in chunks]
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]

    lib.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"Collection '{doc_id}' built with {lib.count()} entries.")
    return lib


def retrieve(query: str, doc_id: str, k: int = 5) -> list[str]:
    lib = _client.get_collection(doc_id)
    embed_q = embed_model.encode([query]).tolist()
    result = lib.query(query_texts=query, query_embeddings=embed_q, n_results=k)
    return result["documents"][0]


def collection_exists(doc_id: str) -> bool:
    try:
        _client.get_collection(doc_id)
        return True
    except Exception:
        return False
