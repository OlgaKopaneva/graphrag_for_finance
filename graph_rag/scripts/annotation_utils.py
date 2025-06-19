import json
import pickle
import networkx as nx
import argparse
from typing import List, Dict, Set, Tuple, Any
import re
from pathlib import Path
from collections import defaultdict


class OntologyAnnotator:
    """
    A class for annotating chunks using an ontology
    """
    def __init__(self, graph_path: str):
        self.graph = self._load_graph(graph_path)
        self.entity_patterns = self._build_entity_patterns()
        self.node_metadata = self._extract_node_metadata()
        
    def _load_graph(self, graph_path: str) -> nx.DiGraph:
        with open(graph_path, 'rb') as f:
            return pickle.load(f)
    
    def _build_entity_patterns(self) -> Dict[str, List[str]]:
        """
        Creates patterns to search for entities in text
        """
        patterns = defaultdict(list)
        
        for node, data in self.graph.nodes(data=True):
            patterns[node.lower()].append(node)
            
            if data.get('comment'):
                comment_words = re.findall(r'\b[A-Za-z]+\b', data['comment'])
                for word in comment_words:
                    if len(word) > 3: 
                        patterns[word.lower()].append(node)
            
            # extracting terms from URIs
            uri = data.get('uri', '')
            if uri:
                if '#' in uri:
                    term = uri.split('#')[-1]
                elif '/' in uri:
                    term = uri.split('/')[-1]
                else:
                    term = uri
                camel_parts = re.findall(r'[A-Z][a-z]*', term)
                for part in camel_parts:
                    if len(part) > 3:
                        patterns[part.lower()].append(node)
        
        return dict(patterns)
    
    def _extract_node_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Retrieves node metadata for annotations"""
        metadata = {}
        
        for node, data in self.graph.nodes(data=True):
            metadata[node] = {
                'uri': data.get('uri', ''),
                'comment': data.get('comment', ''),
                'definition': data.get('definition', ''),
                'node_type': data.get('node_type', 'Class'),
                'source_file': data.get('source_file', ''),
                'neighbors': list(self.graph.neighbors(node))[:5],  
                'predecessors': list(self.graph.predecessors(node))[:5]
            }
        
        return metadata
    
    def annotate_text(self, text: str, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Annotates text, finding mentions of entities from ontology
        """
        text_lower = text.lower()
        annotations = []
        found_entities = set()
        
        # search for entities by patterns
        for pattern, entities in self.entity_patterns.items():
            if pattern in text_lower and len(pattern) > 3:
                for entity in entities:
                    if entity not in found_entities:
                        confidence = self._calculate_confidence(
                            pattern, entity, text, text_lower
                        )
                        
                        if confidence >= min_confidence:
                            annotation = {
                                'entity': entity,
                                'pattern': pattern,
                                'confidence': confidence,
                                'type': self.node_metadata[entity]['node_type'],
                                'uri': self.node_metadata[entity]['uri'],
                                'context': self._extract_context(text, pattern)
                            }
                            annotations.append(annotation)
                            found_entities.add(entity)
        
        return annotations
    
    def _calculate_confidence(self, pattern: str, entity: str, 
                            original_text: str, text_lower: str) -> float:
        """
        Calculates the confidence of an annotation based on various factors
        """
        confidence = 0.0
        
        # Basic confidence 
        confidence += min(len(pattern) / 20.0, 0.3)
        
        # Bonus for exact match with entity name
        if pattern == entity.lower():
            confidence += 0.3
               
        # Penalty for very short patterns
        if len(pattern) < 4:
            confidence -= 0.2
        
        return min(confidence, 1.0)
    
    
    def _extract_context(self, text: str, pattern: str, window: int = 50) -> str:
        text_lower = text.lower()
        pattern_pos = text_lower.find(pattern)
        
        if pattern_pos == -1:
            return ""
        
        start = max(0, pattern_pos - window)
        end = min(len(text), pattern_pos + len(pattern) + window)
        
        return text[start:end].strip()
    
    def annotate_chunks(self, chunks: List[Dict[str, Any]], 
                       min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        annotated_chunks = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            
            if not text:
                annotated_chunks.append(chunk)
                continue
            
            annotations = self.annotate_text(text, min_confidence)[:5]
        
            annotated_chunk = {
                'text': text,
                'metadata': chunk.get('metadata', {}),
                'annotations': annotations,
                'annotation_stats': {
                    'total_annotations': len(annotations),
                    'unique_entities': len(set(ann['entity'] for ann in annotations)),
                    'avg_confidence': sum(ann['confidence'] for ann in annotations) / len(annotations) if annotations else 0.0
                }
            }
            
            if 'chunk_id' not in annotated_chunk['metadata']:
                annotated_chunk['metadata']['chunk_id'] = i
            
            annotated_chunks.append(annotated_chunk)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} chunks...")
        
        return annotated_chunks


def create_annotated_chunks_from_ontology(chunks_path: str,
                                         graph_path: str,
                                         output_path: str,
                                         min_confidence: float = 0.5):

    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loading ontology from {graph_path}...")
    annotator = OntologyAnnotator(graph_path)
    
    print(f"Annotating {len(chunks)} chunks (min_confidence={min_confidence})...")
    annotated_chunks = annotator.annotate_chunks(chunks, min_confidence)
    
    total_annotations = sum(len(chunk.get('annotations', [])) for chunk in annotated_chunks)
    chunks_with_annotations = sum(1 for chunk in annotated_chunks if chunk.get('annotations'))
    
    print(f"  Total chunks: {len(annotated_chunks)}")
    print(f"  Chunks with annotations: {chunks_with_annotations}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Average annotations per chunk: {total_annotations / len(annotated_chunks):.2f}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotated_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Annotated chunks saved to {output_path}")
    
    return annotated_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ontology-based chunk annotation")
    parser.add_argument("--chunks", required=True, help="Path to chunks.json")
    parser.add_argument("--graph", required=True, help="Path to knowledge graph (.gpickle)")
    parser.add_argument("--output", required=True, help="Output path for annotated chunks")
    parser.add_argument("--min-confidence", type=float, default=0.5, 
                       help="Minimum confidence for annotations")
    
    args = parser.parse_args()

    create_annotated_chunks_from_ontology(
        args.chunks, args.graph, args.output, args.min_confidence
    )