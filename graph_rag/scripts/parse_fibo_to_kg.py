import os
import json
import pickle
import networkx as nx
from pathlib import Path
from typing import List, Dict, Union
from rdflib import Graph, URIRef, RDFS, OWL, RDF, Literal, BNode


def get_label_from_graph(g: Graph, uri: Union[URIRef, BNode]) -> str:
    """
    Gets the label from the RDF graph or uses the last part of the URI
    """
    if isinstance(uri, BNode):
        return str(uri)
    
    for label_obj in g.objects(uri, RDFS.label):
        if isinstance(label_obj, Literal):
            return str(label_obj)
    
    uri_str = str(uri)
    if '#' in uri_str:
        return uri_str.split('#')[-1]
    elif '/' in uri_str:
        return uri_str.split('/')[-1]
    
    return uri_str

def add_node_to_graph(g: Graph, uri: URIRef, G: nx.DiGraph, file_path: str, node_type: str = "Class"):
    """
    add node to the graph with all its properties
    """
    label = get_label_from_graph(g, uri)
    
    for label_obj in g.objects(uri, RDFS.label):
        if isinstance(label_obj, Literal):
            label = str(label_obj)
            break
    
    comment = None
    for comment_obj in g.objects(uri, RDFS.comment):
        if isinstance(comment_obj, Literal):
            comment = str(comment_obj)
            break
    
    definition = None
    skos_definition = URIRef("http://www.w3.org/2004/02/skos/core#definition") #SKOS (Simple Knowledge Organization System) - standart
    for def_obj in g.objects(uri, skos_definition):
        if isinstance(def_obj, Literal):
            definition = str(def_obj)
            break

    if not G.has_node(label):
        G.add_node(label, 
                  uri=str(uri),
                  comment=comment,
                  definition=definition,
                  node_type=node_type,
                  source_file=os.path.basename(file_path))

def parse_rdf_file_to_graph(file_path: str, G: nx.DiGraph) -> Dict[str, int]:
    """
    Parses an RDF file and adds nodes and links to the NetworkX graph
    """
    g = Graph()
    
    try:
        if file_path.endswith('.rdf'):
            g.parse(file_path, format="xml")
        elif file_path.endswith('.ttl'):
            g.parse(file_path, format="turtle")
        elif file_path.endswith('.owl'):
            g.parse(file_path, format="xml")
        else:
            g.parse(file_path) #auto
            
    except Exception as e:
        print(f"Parsing error {file_path}: {e}")
        return {"nodes": 0, "edges": 0}

    stats = {"nodes": 0, "edges": 0}

    important_predicates = {
        RDFS.subClassOf: "subClassOf",
        RDFS.subPropertyOf: "subPropertyOf", 
        OWL.equivalentClass: "equivalentClass",
        OWL.equivalentProperty: "equivalentProperty",
        OWL.disjointWith: "disjointWith",
        OWL.inverseOf: "inverseOf",
        OWL.sameAs: "sameAs",
        RDFS.domain: "domain",
        RDFS.range: "range"
        }

    for s in g.subjects(RDF.type, OWL.Class):
        if not isinstance(s, URIRef):
            continue
        
        # add nodes for classes
        add_node_to_graph(g, s, G, file_path)
        stats["nodes"] += 1
        
    # add nodes for properties
    for s in g.subjects(RDF.type, OWL.ObjectProperty):
        if not isinstance(s, URIRef):
            continue
        add_node_to_graph(g, s, G, file_path, node_type="ObjectProperty")
        stats["nodes"] += 1
    
    for s in g.subjects(RDF.type, OWL.DatatypeProperty):
        if not isinstance(s, URIRef):
            continue
        add_node_to_graph(g, s, G, file_path, node_type="DatatypeProperty")
        stats["nodes"] += 1

    # add relations
    for predicate, relation_type in important_predicates.items():
        for s, o in g.subject_objects(predicate):
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                source_label = get_label_from_graph(g, s)
                target_label = get_label_from_graph(g, o)
                
                if not G.has_node(source_label):
                    add_node_to_graph(g, s, G, file_path)
                
                if not G.has_node(target_label):
                    add_node_to_graph(g, o, G, file_path)
                
                if not G.has_edge(source_label, target_label):
                    G.add_edge(source_label, target_label, type=relation_type)
                    stats["edges"] += 1
                        
    system_namespaces = {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/2002/07/owl#",
        "http://www.w3.org/2004/02/skos/core#"
    }
    
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
            if any(str(p).startswith(ns) for ns in system_namespaces):
                continue
            
            source_label = get_label_from_graph(g, s)
            target_label = get_label_from_graph(g, o)
            predicate_label = get_label_from_graph(g, p)
            
            if not G.has_node(source_label):
                add_node_to_graph(g, s, G, file_path)
            
            if not G.has_node(target_label):
                add_node_to_graph(g, o, G, file_path)
            
            if not G.has_edge(source_label, target_label):
                G.add_edge(source_label, target_label, type=predicate_label)
                stats["edges"] += 1

    return stats


def find_rdf_files(ontology_root: str) -> List[str]:
    """
    Finds all RDF files in a folder and subfolders (folder traversal)
    """
    rdf_files = []
    ontology_path = Path(ontology_root)
    
    for ext in ['*.rdf', '*.owl', '*.ttl']:
        rdf_files.extend(ontology_path.rglob(ext))
    
    return [str(f) for f in rdf_files]


def build_knowledge_graph(ontology_root: str = "ontology") -> nx.DiGraph:
    """
    Builds a knowledge graph
    """
    print(f"Create a graph of {ontology_root}...")
    
    G = nx.DiGraph()
    
    rdf_files = find_rdf_files(ontology_root)
    
    total_nodes = 0
    total_edges = 0
    
    for i, file_path in enumerate(rdf_files, 1):
        stats = parse_rdf_file_to_graph(file_path, G)
        total_nodes += stats["nodes"]
        total_edges += stats["edges"]
        print(f"  Added nodes : {stats['nodes']}, relations: {stats['edges']}")
    
    print(f"  Nodes in the graph: {G.number_of_nodes()}")
    print(f"  Relations in the graph: {G.number_of_edges()}")
    
    return G

def save_knowledge_graph(graph: nx.DiGraph, output_path: str = "fibo_knowledge_graph.gpickle"):
    """
    Saves knowledge graph in gpickle format 
    """
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
    print(f"The graph saved")

def save_knowledge_graph_json(graph: nx.DiGraph, output_path: str = "fibo_knowledge_graph.json"):
    """
    Saves knowledge graph in JSON 
    """
    graph_data = {
        "nodes": [
            {
                "id": node,
                "label": node,
                "uri": data.get("uri", ""),
                "comment": data.get("comment", ""),
                "definition": data.get("definition", ""),
                "source_file": data.get("source_file", "")
            }
            for node, data in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                "type": data.get("type", "related")
            }
            for source, target, data in graph.edges(data=True)
        ]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"The graph saved")
    
def load_knowledge_graph(file_path: str) -> nx.DiGraph:
    """
    Loads knowledge graph from gpickle format
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def analyze_graph(graph: nx.DiGraph):
    edge_types = {}
    for _, _, data in graph.edges(data=True):
        edge_type = data.get('type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print(f"\nTypes of relations:")
    for edge_type, count in sorted(edge_types.items()):
        print(f"  {edge_type}: {count}")
    
    node_types = {}
    for _, data in graph.nodes(data=True):
        node_type = data.get('node_type', 'Class')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"\nTypes of nodes:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")
    

if __name__ == "__main__":
    knowledge_graph = build_knowledge_graph("data\\ontology")
    analyze_graph(knowledge_graph)
    save_knowledge_graph(knowledge_graph, "data\\fibo_knowledge_graph.gpickle")
    
    print("\nDone!")
    
