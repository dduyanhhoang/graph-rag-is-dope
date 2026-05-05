"""
GraphRAG Implementation - Dev A
Using NetworkX for graph building and BFS traversal.
"""

import networkx as nx
from typing import List, Dict, Any, Tuple, Set
import time
from collections import deque

from shared import (
    llm_generate,
    llm_generate_json,
    load_corpus,
    load_benchmark_questions,
    save_results_csv,
    load_benchmark_questions
)
from shared.embedder import UniversalEmbedder


# ============================================
# Entity & Triple Extraction
# ============================================

def extract_triples_from_document(doc: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Extract (subject, predicate, object) triples from a document using LLM.
    Focuses on companies, people, products, and technologies.
    """
    # Limit content to avoid token limits
    content = doc['content'][:8000]

    prompt = f"""
Extract factual relationships from this text about AI/tech companies.

IMPORTANT RULES:
1. Only extract relationships about: COMPANIES, PEOPLE, PRODUCTS, TECHNOLOGIES
2. Use these predicates ONLY:
   - founded_by (person founded company)
   - co-founded (person co-founded company)
   - developed (company/product developed something)
   - created (person created product)
   - invested_in (entity invested in company)
   - partnered_with (company partnered with company)
   - acquired (company acquired company)
   - uses (company uses product/service)
   - works_at (person works at company)
   - CEO_of (person is CEO of company)
   - based_in (company based in location)
3. Output format: one triple per line as: (subject)|(predicate)|(object)
4. Extract MAXIMUM 30 triples per document
5. Only extract clearly stated facts, not inferred

Document title: {doc['title']}
Document content:
{content}

Examples of good triples:
(Sam Altman)|(co-founded)|(OpenAI)
(OpenAI)|(developed)|(GPT-4)
(Microsoft)|(invested_in)|(OpenAI)
(OpenAI)|(uses)|(Microsoft Azure)
(Satya Nadella)|(CEO_of)|(Microsoft)

Triples (one per line):
"""

    response = llm_generate(prompt, max_tokens=1500)
    triples = []

    for line in response.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Parse format: (subject)|(predicate)|(object)
        if '(' in line and ')|(' in line and line.endswith(')'):
            try:
                # Remove parentheses and split
                line = line.strip('()')
                parts = line.split(')|(')
                if len(parts) == 3:
                    subject = parts[0].strip()
                    predicate = parts[1].strip()
                    obj = parts[2].strip()
                    triples.append((subject, predicate, obj))
            except:
                continue

    logger.info(f"Extracted {len(triples)} triples from '{doc['title']}'")
    return triples


# ============================================
# Graph Building
# ============================================

class KnowledgeGraph:
    """Knowledge graph using NetworkX with optional Neo4j export."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.embedder = UniversalEmbedder()
        self.node_embeddings = {}  # node -> embedding

    def add_triple(self, subject: str, predicate: str, obj: str, source_doc: str):
        """Add a triple to the graph with metadata."""
        # Add nodes if they don't exist
        for node in [subject, obj]:
            if not self.graph.has_node(node):
                self.graph.add_node(node, type="entity", sources=set())

        # Track source documents
        self.graph.nodes[subject]['sources'].add(source_doc)
        self.graph.nodes[obj]['sources'].add(source_doc)

        # Add edge with relation and source
        self.graph.add_edge(
            subject, obj,
            relation=predicate,
            source_doc=source_doc,
            weight=1.0
        )

    def build_from_documents(self, documents: List[Dict[str, Any]]) -> 'KnowledgeGraph':
        """Build knowledge graph from documents."""
        logger.info(f"Building knowledge graph from {len(documents)} documents...")

        for doc in documents:
            triples = extract_triples_from_document(doc)

            for subject, predicate, obj in triples:
                self.add_triple(subject, predicate, obj, doc['doc_id'])

        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self

    def compute_node_embeddings(self):
        """Generate embeddings for all nodes (for hybrid search)."""
        logger.info("Computing node embeddings...")
        nodes = list(self.graph.nodes())
        texts = [f"{node}" for node in nodes]

        embeddings = self.embedder.embed_documents(texts)
        self.node_embeddings = dict(zip(nodes, embeddings))

        logger.info(f"Computed embeddings for {len(nodes)} nodes")
        return self

    def get_stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_directed": self.graph.is_directed(),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            "components": nx.number_weakly_connected_components(self.graph)
        }

    def export_to_gexf(self, filename: str = "data/graph/knowledge_graph.gexf"):
        """Export graph to GEXF format for visualization."""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Convert sets to lists for serialization
        for node, attrs in self.graph.nodes(data=True):
            if 'sources' in attrs:
                attrs['sources'] = list(attrs['sources'])

        nx.write_gexf(self.graph, filename)
        logger.info(f"Graph exported to {filename}")

    # Optional: Neo4j integration
    def export_to_neo4j(self, uri: str = None, username: str = "neo4j", password: str = None):
        """Export graph to Neo4j (optional - requires neo4j package)."""
        try:
            from neo4j import GraphDatabase
        except ImportError:
            logger.warning("Neo4j driver not installed. Skipping Neo4j export.")
            return

        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        password = password or os.getenv("NEO4J_PASSWORD", "password")

        driver = GraphDatabase.driver(uri, auth=(username, password))

        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")

            # Create nodes
            for node, attrs in self.graph.nodes(data=True):
                session.run(
                    "CREATE (n:Entity {name: $name, type: $type, sources: $sources})",
                    name=node,
                    type=attrs.get('type', 'entity'),
                    sources=list(attrs.get('sources', []))
                )

            # Create relationships
            for u, v, attrs in self.graph.edges(data=True):
                session.run(
                    "MATCH (a:Entity {name: $u}), (b:Entity {name: $v}) "
                    "CREATE (a)-[r:RELATION {type: $relation, source_doc: $source_doc}]->(b)",
                    u=u, v=v,
                    relation=attrs.get('relation', 'related'),
                    source_doc=attrs.get('source_doc', 'unknown')
                )

        driver.close()
        logger.info("Graph exported to Neo4j")


# ============================================
# Graph Retrieval (BFS)
# ============================================

def find_seed_nodes(entities: List[str], graph: nx.DiGraph, max_matches: int = 3) -> List[str]:
    """
    Find nodes in graph that match the extracted entities.
    Uses fuzzy matching for flexibility.
    """
    matches = []

    for entity in entities:
        entity_lower = entity.lower()
        entity_norm = normalize_entity_name(entity)

        # Exact match
        if entity in graph.nodes():
            matches.append(entity)
            continue

        # Case-insensitive substring match
        for node in graph.nodes():
            node_lower = node.lower()
            if entity_lower in node_lower or node_lower in entity_lower:
                matches.append(node)
                if len(matches) >= max_matches:
                    break

        # Normalized match
        if not any(entity_norm == normalize_entity_name(m) for m in matches):
            for node in graph.nodes():
                if normalize_entity_name(node) == entity_norm:
                    matches.append(node)
                    break

    return list(set(matches))[:max_matches]


def bfs_traverse(
    graph: nx.DiGraph,
    seed_nodes: List[str],
    depth: int = 2,
    max_edges: int = 50,
    include_edges: bool = True
) -> List[str]:
    """
    Perform BFS traversal from seed nodes and collect subgraph.
    Returns list of triple strings: "(subject)-[predicate]->(object)"
    """
    visited = set()
    edges_collected = []
    queue = deque([(node, 0) for node in seed_nodes])

    while queue and len(edges_collected) < max_edges:
        current, current_depth = queue.popleft()

        if current in visited:
            continue
        visited.add(current)

        if current_depth >= depth:
            continue

        # Get outgoing edges
        for neighbor in graph.neighbors(current):
            edge_data = graph.get_edge_data(current, neighbor)
            relation = edge_data.get('relation', 'related_to')
            edges_collected.append(f"({current})-[{relation}]->({neighbor})")

            if neighbor not in visited:
                queue.append((neighbor, current_depth + 1))

        # Also get incoming edges for context
        for predecessor in graph.predecessors(current):
            edge_data = graph.get_edge_data(predecessor, current)
            relation = edge_data.get('relation', 'related_to')
            edges_collected.append(f"({predecessor})-[{relation}]->({current})")

            if predecessor not in visited:
                queue.append((predecessor, current_depth + 1))

    return edges_collected[:max_edges]


def textualize_subgraph(triples: List[str]) -> str:
    """Convert list of triples to natural language text for LLM."""
    if not triples:
        return "No relationships found."

    text = "Knowledge Graph Relationships:\n\n"
    for i, triple in enumerate(triples, 1):
        text += f"{i}. {triple}\n"

    return text


# ============================================
# GraphRAG Query Pipeline
# ============================================

class GraphRAG:
    """Complete GraphRAG system."""

    def __init__(
        self,
        graph: KnowledgeGraph = None,
        traversal_depth: int = 2,
        max_edges: int = 50
    ):
        self.graph = graph or KnowledgeGraph()
        self.traversal_depth = traversal_depth
        self.max_edges = max_edges

    def query(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Execute GraphRAG query pipeline.
        Returns (answer, metadata)
        """
        start_time = time.time()
        metadata = {
            "entities": [],
            "seed_nodes": [],
            "triples_count": 0,
            "traversal_depth": self.traversal_depth
        }

        # 1. Extract entities from question
        entities = extract_entities_from_question(question)
        metadata["entities"] = entities

        if not entities:
            return "I couldn't identify any entities in your question.", metadata

        # 2. Find seed nodes in graph
        seed_nodes = find_seed_nodes(entities, self.graph.graph)
        metadata["seed_nodes"] = seed_nodes

        if not seed_nodes:
            return f"I couldn't find information about {', '.join(entities)} in the knowledge graph.", metadata

        # 3. BFS traversal
        triples = bfs_traverse(
            self.graph.graph,
            seed_nodes,
            depth=self.traversal_depth,
            max_edges=self.max_edges
        )
        metadata["triples_count"] = len(triples)

        if not triples:
            return "No relationships found in the knowledge graph.", metadata

        # 4. Textualization
        context = textualize_subgraph(triples)

        # 5. Generate answer
        prompt = f"""
Answer the question using ONLY the relationships provided in the knowledge graph.

Rules:
- Base your answer ONLY on the provided relationships
- If information is insufficient, say "I don't have enough information"
- Do not make up facts
- Cite the relationship numbers when possible

Question: {question}

{context}

Answer:
"""

        answer = llm_generate(prompt, max_tokens=500)
        latency = int((time.time() - start_time) * 1000)
        metadata["latency_ms"] = latency

        return answer, metadata


def build_graphrag_system(documents: List[Dict[str, Any]]) -> GraphRAG:
    """Build complete GraphRAG system from documents."""
    logger.info("Building GraphRAG system...")

    # Build knowledge graph
    kg = KnowledgeGraph()
    kg.build_from_documents(documents)
    kg.compute_node_embeddings()

    # Create GraphRAG system
    graphrag = GraphRAG(graph=kg)

    logger.info(f"GraphRAG ready: {kg.get_stats()}")
    return graphrag


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("GRAPHRAG SYSTEM TEST (Dev A)")
    print("="*60)

    # Load corpus
    documents = load_corpus()
    print(f"Loaded {len(documents)} documents")

    if len(documents) == 0:
        print("ERROR: No documents loaded. Check corpus/ directory.")
        exit(1)

    # Build system
    graphrag = build_graphrag_system(documents[:min(10, len(documents))])  # Test with first 10

    # Export graph
    graphrag.graph.export_to_gexf("data/graph/knowledge_graph.gexf")
    print("Graph exported to data/graph/knowledge_graph.gexf")

    # Test queries
    test_questions = [
        "What relationship does Microsoft have with OpenAI?",
        "Which companies invested in OpenAI?",
        "Who founded Anthropic?"
    ]

    print("\n" + "="*60)
    print("TEST QUERIES")
    print("="*60)

    for question in test_questions:
        print(f"\nQ: {question}")
        answer, metadata = graphrag.query(question)
        print(f"A: {answer[:200]}...")
        print(f"  Entities: {metadata['entities']}")
        print(f"  Seed nodes: {metadata['seed_nodes']}")
        print(f"  Triples: {metadata['triples_count']}")
        print(f"  Latency: {metadata['latency_ms']}ms")

    print("\n" + "="*60)
    print("GraphRAG test complete!")
    print("="*60)
