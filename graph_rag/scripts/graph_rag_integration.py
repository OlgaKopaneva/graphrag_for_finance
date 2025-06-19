import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from graph_rag.core.retrievers.graph_rag_retriever import GraphRAGRetriever, create_graph_rag_retriever
from graph_rag.core.retrievers.hybrid_retriever import build_hybrid_retriever
from graph_rag.config.config import GRAPH_PATH, ANNOTATED_CHUNKS_PATH, CHUNKS_PATH, GRAPH_WEIGHT, DEFAULT_TOP_K, EXPAND_NEIGHBORS, MAX_NEIGHBOR_DEPTH


class GraphRAGPipeline:
    def __init__(self):
        self.graph_path = str(GRAPH_PATH)
        self.annotated_chunks_path = str(ANNOTATED_CHUNKS_PATH)
        self.chunks_path = str(CHUNKS_PATH)
        self.graph_weight = GRAPH_WEIGHT
        self.default_top_k = DEFAULT_TOP_K
        self.expand_neighbors = EXPAND_NEIGHBORS
        self.max_neighbor_depth = MAX_NEIGHBOR_DEPTH
        self.graph_rag_retriever: Optional[GraphRAGRetriever] = None
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialises the GraphRAG retriever"""
        try:
            if not Path(self.graph_path).exists():
                raise FileNotFoundError(f"Graph file not found: {self.graph_path}")
            if not Path(self.annotated_chunks_path).exists():
                raise FileNotFoundError(f"Annotated chunks file not found: {self.annotated_chunks_path}")
        
            hybrid_retriever = build_hybrid_retriever()
            if hybrid_retriever is None:
                raise ValueError("Failed to initialize hybrid retriever")
            
            self.graph_rag_retriever = create_graph_rag_retriever(
                graph_path = self.graph_path,
                annotated_chunks_path=self.annotated_chunks_path,
                hybrid_retriever=hybrid_retriever,
                graph_weight=self.graph_weight
            )
            print("GraphRAG retriever initialized successfully")
            
        except Exception as e:
            print(f"Error initializing GraphRAG retriever: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            self.graph_rag_retriever = None
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Performs a search via GraphRAG
        """
        if not self.graph_rag_retriever:
            raise RuntimeError("GraphRAG retriever not initialized")
        
        if top_k is None:
            top_k = self.default_top_k
            
        return self.graph_rag_retriever.retrieve(query, top_k=top_k)
    
    def format_results(self, results: List[Dict[str, Any]], show_metadata: bool = True) -> str:
        """
        Formats the results for output
        """
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            text = result.get('text', 'No text available')
            source = result.get('retrieval_source', 'unknown')
            weight = result.get('retrieval_weight', 0.0)
            
            result_text = f"\n--- Result {i} ---\n"
            
            if show_metadata:
                result_text += f"Source: {source} (weight: {weight:.2f})\n"
                if 'source_file' in result:
                    result_text += f"File: {result['source_file']}\n"
            
            if len(text) > 500:
                result_text += text[:500] + "...\n"
            else:
                result_text += text + "\n"
                
            formatted.append(result_text)
        
        return "\n".join(formatted)


def main():
    parser = argparse.ArgumentParser(description="GraphRAG Pipeline")
    parser.add_argument("--query", type=str, 
                       help="Search query")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of results to return")
    parser.add_argument("--stats", action="store_true",
                       help="Show system statistics")
    
    args = parser.parse_args()
    
    try:
        pipeline = GraphRAGPipeline()
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return
    
    if args.query:
        try:
            print(f"Searching for: '{args.query}'")
            print("=" * 50)
            
            results = pipeline.search(args.query, top_k=args.top_k)
            formatted_results = pipeline.format_results(results, show_metadata=True)
            
            print(formatted_results)
            print(f"\nTotal results: {len(results)}")
            
        except Exception as e:
            print(f"Error during search: {e}")
        return


if __name__ == "__main__":
    main()