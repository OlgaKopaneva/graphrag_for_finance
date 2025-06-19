from pathlib import Path

DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "indexes"
CHUNKS_PATH = DATA_DIR / "chunks.json"
GRAPH_PATH = DATA_DIR / "fibo_knowledge_graph.gpickle"
ANNOTATED_CHUNKS_PATH = DATA_DIR / "annotated_chunks.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BM25_K = 10
FAISS_K = 10
DEFAULT_TOP_K = 10
ENSEMBLE_WEIGHTS = [0.5, 0.5]  # [BM25, FAISS]
GRAPH_WEIGHT = 0.7
EXPAND_NEIGHBORS = True
MAX_NEIGHBOR_DEPTH = 1