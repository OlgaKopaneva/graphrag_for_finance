import json
import pickle
import networkx as nx
from typing import List, Dict, Tuple, Any
from pathlib import Path
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class GraphRetriever:
    """
    Class for finding relevant nodes in a knowledge graph
    """
    def __init__(self, graph_path: str):
        self.graph = self._load_graph(graph_path)
        self.node_to_text = self._build_node_text_mapping()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.node_vectors = self._build_node_vectors()
        
    def _load_graph(self, graph_path: str) -> nx.DiGraph:
        """Loads the knowledge graph"""
        with open(graph_path, 'rb') as f:
            return pickle.load(f)
    
    def _build_node_text_mapping(self) -> Dict[str, str]:
        """Creates a textual representation of each node for searching"""
        node_to_text = {}
        
        for node, data in self.graph.nodes(data=True):
            text_parts = [node]  
            
            if data.get('comment'):
                text_parts.append(data['comment'])
            if data.get('definition'):
                text_parts.append(data['definition'])
                
            neighbors = list(self.graph.neighbors(node))
            predecessors = list(self.graph.predecessors(node))
            
            if neighbors:
                text_parts.append(" ".join(neighbors[:5]))  
            if predecessors:
                text_parts.append(" ".join(predecessors[:5]))
                
            node_to_text[node] = " ".join(text_parts)
            
        return node_to_text
    
    def _build_node_vectors(self) -> spmatrix:
        """Creates vector representations of nodes"""
        node_texts = list(self.node_to_text.values())
        return self.vectorizer.fit_transform(node_texts)
    
    def find_relevant_nodes(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Finds the most relevant nodes for the query
        """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.node_vectors).flatten()
        node_names = list(self.node_to_text.keys())
        node_similarities = [(node_names[i], similarities[i]) for i in range(len(node_names))]
        node_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return node_similarities[:top_k]
    
    def expand_with_neighbors(self, nodes: List[str], max_depth: int = 2) -> List[str]:
        """
        Expands the list of nodes with their neighbours in the graph
        """
        expanded_nodes = set(nodes)
        
        for depth in range(max_depth):
            current_level = set()
            for node in expanded_nodes:
                if node in self.graph:
                    current_level.update(self.graph.neighbors(node))
                    current_level.update(self.graph.predecessors(node))
            expanded_nodes.update(current_level)
            
            if len(expanded_nodes) > 50:
                break
                
        return list(expanded_nodes)


class AnnotatedChunkRetriever:
    """
    Сlass for searching annotated chunks by graph nodes
    """
    
    def __init__(self, annotated_chunks_path: str):
        self.annotated_chunks = self._load_annotated_chunks(annotated_chunks_path)
        self.node_to_chunks = self._build_node_chunk_mapping()
        
    def _load_annotated_chunks(self, path: str) -> List[Dict]:
        """Loads annotated chunks"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_node_chunk_mapping(self) -> Dict[str, List[int]]:
        """Creates mapping from nodes to chunk indexes"""
        node_to_chunks = {}
        
        for idx, chunk in enumerate(self.annotated_chunks):
            annotations = chunk.get('annotations', []) or chunk.get('entities', [])
            
            for annotation in annotations:
                if isinstance(annotation, str):
                    node_name = annotation
                elif isinstance(annotation, dict):
                    node_name = annotation.get('entity', annotation.get('name', ''))
                else:
                    continue
                    
                if node_name:
                    if node_name not in node_to_chunks:
                        node_to_chunks[node_name] = []
                    node_to_chunks[node_name].append(idx)
                    
        return node_to_chunks
    
    def get_chunks_by_nodes(self, nodes: List[str]) -> List[Dict]:
        """
        Gets the chunks associated with the specified nodes
        """
        chunk_indices = set()
        
        for node in nodes:
            if node in self.node_to_chunks:
                chunk_indices.update(self.node_to_chunks[node])
        
        return [self.annotated_chunks[idx] for idx in chunk_indices]


class GraphRAGRetriever:
    """
    GraphRAG main class that combines knowledge graph and hybrid search
    """
    def __init__(self, 
                 graph_path: str,
                 annotated_chunks_path: str,
                 hybrid_retriever,
                 graph_weight: float = 0.7):
        """
        graph_path: path to the knowledge graph file
        annotated_chunks_path: path to annotated chunks
        hybrid_retriever: (FAISS+BM25)
        graph_weight: weighting of graph results (0.0-1.0)
        """
        self.graph_retriever = GraphRetriever(graph_path)
        self.chunk_retriever = AnnotatedChunkRetriever(annotated_chunks_path)
        self.hybrid_retriever = hybrid_retriever
        self.graph_weight = graph_weight
        
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        # Search for relevant nodes in graph
        relevant_nodes = self.graph_retriever.find_relevant_nodes(query, top_k=10)
        node_names = [node for node, score in relevant_nodes]
        
        # Extension of nodes by neighbours
        expanded_nodes = self.graph_retriever.expand_with_neighbors(node_names, max_depth=1)
        
        # Obtaining annotated chunks by graph nodes
        graph_chunks = self.chunk_retriever.get_chunks_by_nodes(expanded_nodes)
        
        # Hybrid search (FAISS+BM25)
        hybrid_results = self.hybrid_retriever.get_relevant_documents(query)
        hybrid_chunks = [{"text": doc.page_content, "source": "hybrid"} for doc in hybrid_results]
        
        # Combining results with weights
        combined_results = self._combine_results(graph_chunks, hybrid_chunks, top_k)
        
        return combined_results
    
    def _combine_results(self, 
                        graph_chunks: List[Dict], 
                        hybrid_chunks: List[Dict], 
                        top_k: int) -> List[Dict]:

        graph_count = int(top_k * self.graph_weight)
        hybrid_count = top_k - graph_count
        
        selected_graph = graph_chunks[:graph_count]
        selected_hybrid = hybrid_chunks[:hybrid_count]
        
        for chunk in selected_graph:
            chunk['retrieval_source'] = 'graph'
            chunk['retrieval_weight'] = self.graph_weight
            
        for chunk in selected_hybrid:
            chunk['retrieval_source'] = 'hybrid'
            chunk['retrieval_weight'] = 1.0 - self.graph_weight
        
        combined = selected_graph + selected_hybrid
        
        seen_texts = set()
        unique_results = []
        
        for chunk in combined:
            chunk_text = chunk.get('text', '')[:100]  
            if chunk_text not in seen_texts:
                seen_texts.add(chunk_text)
                unique_results.append(chunk)
        
        return unique_results[:top_k]


def create_graph_rag_retriever(graph_path: str,
                              annotated_chunks_path: str,
                              hybrid_retriever,
                              graph_weight: float = 0.7) -> GraphRAGRetriever:
    return GraphRAGRetriever(
        graph_path=graph_path,
        annotated_chunks_path=annotated_chunks_path,
        hybrid_retriever=hybrid_retriever,
        graph_weight=graph_weight
    )
