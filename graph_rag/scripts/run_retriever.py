import sys
import os
import argparse
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from graph_rag.core.retrievers import bm25_retriever, faiss_retriever, hybrid_retriever
from graph_rag.core.retrievers.graph_rag_retriever import GraphRAGRetriever, create_graph_rag_retriever
from graph_rag.scripts.graph_rag_integration import GraphRAGPipeline
from graph_rag.config.config import CHUNKS_PATH, INDEX_DIR, FAISS_K, ENSEMBLE_WEIGHTS

class EnhancedRetrieverPipeline:
    """
    Advanced search pipelines combining regular RAG and GraphRAG approaches
    """
    def __init__(self, 
                 use_graph_rag: bool = True):
        self.use_graph_rag = use_graph_rag
        self.regular_retriever = None
        self.graph_rag_pipeline = None
        
        self._init_regular_retriever()
        
        if use_graph_rag:
            self._init_graph_rag()
    
    def _init_regular_retriever(self):
        """Initialises the traditional hybrid retriever (FAISS+BM25)"""
        try:
            self.regular_retriever = hybrid_retriever.build_hybrid_retriever()
            print("Regular hybrid retriever initialized")
        except Exception as e:
            print(f"Error initializing regular retriever: {e}")
    
    def _init_graph_rag(self):
        """Initialises the GraphRAG pipeline (GraphRAG + (FAISS+BM25))"""
        try:
            self.graph_rag_pipeline = GraphRAGPipeline()
            print("GraphRAG pipeline initialized")
        except Exception as e:
            print(f"Error initializing GraphRAG: {e}")
            self.use_graph_rag = False
    
    def search(self, 
               query: str, 
               k: int = 7, 
               method: str = "graph_rag") -> List[Dict[str, Any]]:
        """
        Performs a search using the selected method
            k: number of results
            method: search method ("regular", "graph_rag")
        """
        if method == "regular" or not self.use_graph_rag:
            return self._search_regular(query, k)
        elif method == "graph_rag" and self.use_graph_rag:
            return self._search_graph_rag(query, k)
        else:
            raise ValueError(f"Unknown search method: {method}")
    
    def _search_regular(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search via traditional hybrid retriever (FAISS+BM25)"""
        if not self.regular_retriever:
            raise RuntimeError("Regular retriever not initialized")
        
        results = self.regular_retriever.get_relevant_documents(query)[:k]
        
        formatted_results = []
        for i, doc in enumerate(results):
            formatted_results.append({
                "text": doc.page_content,
                "metadata": getattr(doc, 'metadata', {}),
                "retrieval_source": "traditional",
                "retrieval_method": "hybrid_faiss_bm25",
                "rank": i + 1,
                "score": getattr(doc, 'score', 0.0)
            })
        
        return formatted_results
    
    def _search_graph_rag(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search via  GraphRAG + traditional hybrid retriever (FAISS+BM25)"""
        if not self.graph_rag_pipeline:
            raise RuntimeError("GraphRAG pipeline not initialized")
        
        results = self.graph_rag_pipeline.search(query, top_k=k)
        
        for i, result in enumerate(results):
            result["rank"] = i + 1
            result["retrieval_method"] = "graph_rag"
        
        return results
    
    
    def compare_methods(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Compares the results of different search methods
        """
        comparison = {}
        
        if self.regular_retriever:
            try:
                comparison["regular"] = self._search_regular(query, k)
            except Exception as e:
                comparison["regular"] = {"error": str(e)}
        
        if self.use_graph_rag and self.graph_rag_pipeline:
            try:
                comparison["graph_rag"] = self._search_graph_rag(query, k)
            except Exception as e:
                comparison["graph_rag"] = {"error": str(e)}
        
        return comparison


def load_texts() -> List[str]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [chunk["text"] for chunk in chunks]


def build_indexes():
    """Builds indexes for regular search"""
    texts = load_texts()
    INDEX_DIR.mkdir(exist_ok=True)
    
    bm25_retrieve = bm25_retriever.build_bm25(texts)
    faiss_index = faiss_retriever.build_faiss(texts)
    
    bm25_retriever.save_bm25(bm25_retrieve)
    faiss_retriever.save_faiss(faiss_index)
    print(f"Traditional indices saved in {INDEX_DIR}")


def print_results(results: List[Dict[str, Any]], method: str = ""):
    if not results:
        print("No results found.")
        return
    
    print(f"\n{'='*60}")
    if method:
        print(f"Results using {method.upper()} method:")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        
        source = result.get('retrieval_source', 'unknown')
        method_used = result.get('retrieval_method', 'unknown')
        print(f"Source: {source} | Method: {method_used}")
        
        if 'score' in result:
            print(f"Score: {result['score']:.4f}")
        if 'retrieval_weight' in result:
            print(f"Weight: {result['retrieval_weight']:.2f}")
        
        text = result.get('text', 'No text available')
        if len(text) > 500:
            print(f"Text: {text[:500]}...")
        else:
            print(f"Text: {text}")


def print_comparison(comparison: Dict[str, Any], query: str):
    print(f"\n{'='*80}")
    print(f"COMPARISON RESULTS FOR QUERY: '{query}'")
    print(f"{'='*80}")
    
    for method, results in comparison.items():
        print(f"\n{'-'*40}")
        print(f"METHOD: {method.upper()}")
        print(f"{'-'*40}")
        
        if isinstance(results, dict) and "error" in results:
            print(f"Error: {results['error']}")
        elif isinstance(results, list):
            print(f"Found {len(results)} results")
            for i, result in enumerate(results[:3], 1):  
                text = result.get('text', 'No text')[:200]
                source = result.get('retrieval_source', 'unknown')
                print(f"  {i}. [{source}] {text}...")
        else:
            print("No results")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Retriever with GraphRAG")
    parser.add_argument("--build", action="store_true", 
                       help="Build regular search indexes")
    parser.add_argument("--query", type=str,
                       help="Search query")
    parser.add_argument("-k", "--top-k", type=int, default=5,
                       help="Number of results to return")
    parser.add_argument("--method", choices=["regular", "graph_rag"], 
                       default="graph_rag",
                       help="Search method to use")
    parser.add_argument("--compare", action="store_true",
                       help="Compare all available methods")
    parser.add_argument("--no-graph", action="store_true",
                       help="Disable GraphRAG (use only traditional)")
    
    args = parser.parse_args()
    
    if args.build:
        build_indexes()
        return
    
    try:
        pipeline = EnhancedRetrieverPipeline(
            use_graph_rag=not args.no_graph
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return
    
    if args.query:
        if args.compare:
            comparison = pipeline.compare_methods(args.query, args.top_k)
            print_comparison(comparison, args.query)
        else:
            results = pipeline.search(args.query, args.top_k, args.method)
            print_results(results, args.method)
        return


if __name__ == "__main__":
    main()

