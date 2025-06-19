from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from graph_rag.config.config import EMBEDDING_MODEL, INDEX_DIR
from pathlib import Path

def build_faiss(texts: list[str]) -> FAISS:
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_texts(texts, embedding)

def save_faiss(index: FAISS):
    index.save_local(str(INDEX_DIR / "faiss"))

def load_faiss() -> FAISS:
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(str(INDEX_DIR / "faiss"), embedding, allow_dangerous_deserialization=True)