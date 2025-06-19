import sys
from graph_rag.core.generation import generate_answer
from graph_rag.scripts.run_retriever import EnhancedRetrieverPipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m graph_rag.scripts.generate_answer_cli \"<your question>\" [method] [top_k]")
        return

    query = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "graph_rag"
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    try:
        pipeline = EnhancedRetrieverPipeline(use_graph_rag=True)
    except Exception as e:
        print(f"Error initializing retriever pipeline: {e}")
        exit(1)
        
    try:
        retrieved = pipeline.search(query, top_k, method)
    except Exception as e:
        print(f"Retrieval failed for question: {query} — {e}")
        retrieved = []

    context = [chunk["text"] for chunk in retrieved]
    response = generate_answer(query, context)
    print("\nAnswer:\n", response)
    
if __name__ == "__main__":
    main()