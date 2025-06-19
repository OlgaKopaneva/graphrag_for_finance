from langchain_community.retrievers import BM25Retriever
from graph_rag.config.config import BM25_K, INDEX_DIR
import pickle

def build_bm25(texts: list[str]) -> BM25Retriever:
    retriever = BM25Retriever.from_texts(texts)
    retriever.k = BM25_K
    return retriever

def save_bm25(retriever: BM25Retriever):
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(retriever, f)

def load_bm25() -> BM25Retriever:
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        return pickle.load(f)