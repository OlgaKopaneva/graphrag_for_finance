from langchain.retrievers import EnsembleRetriever
from graph_rag.core.retrievers.bm25_retriever import load_bm25
from graph_rag.core.retrievers.faiss_retriever import load_faiss
from graph_rag.config.config import ENSEMBLE_WEIGHTS, FAISS_K

def build_hybrid_retriever():
    bm25 = load_bm25()
    faiss = load_faiss().as_retriever(search_kwargs={"k": FAISS_K})
    return EnsembleRetriever(
        retrievers=[bm25, faiss],
        weights=ENSEMBLE_WEIGHTS
    )