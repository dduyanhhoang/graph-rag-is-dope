# GraphRAG & Knowledge Graphs - Lab Summary

## Overview

This lab (AICB-P2T3 Day 19) focuses on building a GraphRAG system that can handle multi-hop relational queries that traditional Flat RAG systems fail on. The key insight: **GraphRAG enables accurate multi-hop reasoning by understanding structural relationships between entities, while Flat RAG only finds semantic similarity.**

---

## The Core Problem: When Flat RAG Fails

Flat RAG struggles with **3 types of queries**:

1. **Multi-hop relational**: "A is linked to B through C" - requires connecting entities across documents
2. **Global thematic**: "What's the overall theme X in the corpus?"
3. **Cross-document comparison**: "Compare policy A with policy B"

### Example: The Hallucination Problem

**Question**: "What AI companies were co-founded by ex-Google employees who worked on Transformer architecture?"

Flat RAG retrieves chunks about Google, Transformers, and AI startups **separately** but **cannot link them** → hallucinates or says "I don't have enough information."

---

## Knowledge Graph Fundamentals

### Core Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Node** (Entity) | A thing/entity in your data | `Sam Altman`, `OpenAI`, `Microsoft` |
| **Edge** (Relation) | Connection between nodes | `CO_FOUNDED`, `DEVELOPED`, `WORKED_AT` |
| **Triple** | Atomic unit: (subject, predicate, object) | `(Sam Altman, CO_FOUNDED, OpenAI)` |
| **Property** | Metadata on nodes/edges | `age=38`, `year=2015` |

**Directed Labeled Graphs**: Knowledge Graphs are almost always directed (edges have direction) and labeled (edges have types).

### Graph Databases Comparison

| Feature | NetworkX | Neo4j |
|---------|----------|-------|
| **Setup** | `pip install` | Docker/Desktop |
| **Scale** | ~100K nodes | Millions+ |
| **Query Language** | Python API | Cypher |
| **Best For** | Prototyping/Research | Production |
| **ACID** | No | Yes |

**Cypher Example**:
```cypher
MATCH (p:Person)-[:CO_FOUNDED]->(c:Company)
WHERE c.name = 'OpenAI'
RETURN p.name
```

---

## The GraphRAG Pipeline

```
Query → Seed Node Matching → Graph Traversal → Textualization → LLM Generation
```

### Step-by-Step Process

1. **Query Processing**: Extract key entities from user question using LLM
2. **Seed Node Matching**: Find matching nodes in graph DB (exact match or semantic similarity)
3. **Graph Traversal**: BFS from seed nodes, collecting connected triples (typical depth = 2 hops)
4. **Textualization**: Convert graph data to text prompt:
   ```
   The following relationships exist in the knowledge base:
   Bill Gates CO_FOUNDED Microsoft. Paul Allen CO_FOUNDED Microsoft.
   ```
5. **Generation**: LLM generates answer **only using** provided context

---

## Extraction Challenges (The Bottleneck)

Building the graph is the hardest part. These steps are critical:

| Step | Purpose | What Happens If Skipped |
|------|---------|------------------------|
| **Coreference Resolution** | Link pronouns to entities | "He founded OpenAI" creates isolated "He" node → loses 30-40% of relationships |
| **Entity Disambiguation** | Separate entities with same name | "Apple" (company) vs "Apple" (fruit) merge incorrectly |
| **Entity Deduplication** | Normalize entity names | "OpenAI", "Open AI", "open-ai" create 3 separate nodes |

**GIGO Principle**: Garbage In = Garbage Out. Graph quality is more important than graph size.

---

## SOTA Architectures

### Microsoft GraphRAG (2024)

**Philosophy**: Sensemaking - understand the big picture of your entire dataset.

**Key Innovation**: Community Detection + Hierarchical Summarization

1. Uses **Leiden algorithm** to detect communities (clusters of tightly-linked nodes)
2. Creates **hierarchical summaries** at every level (pre-computed)
3. Two query modes:
   - **Local Search**: Specific questions → traverse nodes
   - **Global Search**: Thematic questions → read community summaries

**Weaknesses**:
- Very expensive indexing (LLM summarization for all communities)
- 1M tokens can cost $10-50+
- Not suitable for frequently changing data

### LightRAG

**Key Innovation**: Dual-level retrieval without pre-computing expensive summaries.

**How it works**:
- Creates vector embeddings for both **Nodes AND Edges**
- Uses vector search to find relevant nodes AND relationships simultaneously
- Avoids expensive graph traversal computation

---

## Comparison: Flat RAG vs GraphRAG

| Aspect | Flat RAG | MS GraphRAG | LightRAG |
|--------|----------|-------------|----------|
| **Index Cost** | Very Cheap | Very Expensive | Moderate |
| **Query Cost** | Cheap | Moderate | Moderate |
| **Global Understanding** | Poor | Excellent | Good |
| **Multi-hop Accuracy** | Poor | Excellent | Good |
| **Best Use Case** | Single-doc factoids | Thematic analysis | Fast multi-hop |

---

## Decision Framework: Which One to Use?

### Use Flat RAG when:
- ✅ IT support document lookup
- ✅ Basic customer service chatbot
- ✅ Single-document fact retrieval
- ✅ Answer is clearly in one chunk

### Upgrade to GraphRAG when:
- ✅ Investigative journalism / intelligence analysis
- ✅ Complex medical literature review
- ✅ Legal discovery with cross-document relationships
- ✅ Answer requires synthesizing multiple documents
- ✅ **Target: 20%+ multi-hop accuracy improvement**

---

## Hybrid Approach

**GraphRAG doesn't replace Vector Search - they complement each other:**

```
Query → Vector Search (find seed nodes) → Graph Traversal (find relationships) → LLM Generation
```

Store vector embeddings **inside** graph nodes for hybrid capabilities.

---

## Enterprise Use Cases

### 1. Legal & Compliance
Map contract dependencies: See how a regulatory change ripples through vendor contracts.

### 2. HR & Organization Mapping
Internal "company brain": Connect skills, projects, and people.
```
Query: "Find people who know Python AND worked on Project X"
```

### 3. Supply Chain Risk
Physical world mapping: "If Supplier A stops, which products are delayed?"

---

## Lab #19: Build Your Own GraphRAG

### Objectives
- Build Knowledge Graph from Tech Company Corpus
- Implement GraphRAG retrieval pipeline
- Benchmark: GraphRAG vs Flat RAG on multi-hop accuracy
- **Target**: >20% accuracy improvement on multi-hop questions

### Tools & Setup

```bash
# Install dependencies
pip install networkx matplotlib neo4j openai pandas noderag langchain langchain-openai

# For Neo4j: Use Neo4j Desktop or Docker for visualization
```

### Step-by-Step Implementation

#### **Step 1: Entity & Relation Extraction (Indexing)**

Use LLM to convert documents into triples:

```
Input: "OpenAI was founded by Sam Altman and Elon Musk in 2015."

Output triples:
(OpenAI, FOUNDED_BY, Sam Altman)
(OpenAI, FOUNDED_BY, Elon Musk)
(OpenAI, FOUNDED_IN, 2015)
```

#### **Step 2: Build the Graph**

Choose one:
- **NetworkX**: Fast prototyping, offline notebooks
- **Neo4j**: Production, visual exploration (recommended)
- **NodeRAG**: All-in-one framework (already optimized)

Add embeddings to each node for hybrid search capability.

#### **Step 3: Implement Query Pipeline**

```python
def graphrag_query(question):
    # 1. Extract entities from question
    entities = llm_extract_entities(question)  # e.g., ["Google", "Transformer"]

    # 2. Find seed nodes in graph
    seed_nodes = find_nodes(entities)

    # 3. BFS traversal (2-hop recommended)
    subgraph = bfs_traverse(seed_nodes, depth=2, max_edges=50)

    # 4. Textualization
    context = triples_to_text(subgraph)

    # 5. Generate answer
    return llm_generate(question, context)
```

**Traversal Depth Guidelines**:
- `depth=1`: Too shallow, misses multi-hop context
- `depth=2`: Standard recommendation ✅
- `depth=3+`: Too noisy, overwhelms context window

#### **Step 4: Benchmark**

Compare on 20 multi-hop questions:

| Metric | Flat RAG | GraphRAG |
|--------|----------|----------|
| Accuracy | ? | ? |
| Latency | ? | ? |
| Token Cost | ? | ? |

Record cases where Flat RAG hallucinates but GraphRAG is correct.

---

## Deliverables (Due in 2 hours)

1. **Source code** (`.py` or `.ipynb`)
2. **Graph visualization screenshot** (from Neo4j Browser or Matplotlib)
3. **Benchmark comparison table** (20 questions, accuracy/latency/cost)
4. **Failure mode analysis** (why Flat RAG failed on specific queries)

---

## Key Takeaways

1. **Knowledge graphs enable multi-hop reasoning** that flat RAG cannot do
2. **Entity extraction quality is the bottleneck** - invest in NER, coreference resolution, disambiguation
3. **Graph quality > Graph size**: 1,000 high-quality triples beats 100K noisy ones
4. **GraphRAG pipeline is production-ready** - customize entity extraction for your domain

---

## Next Steps

- **Day 20**: Multi-Agent Systems (scale up from single agent to supervisor, debate, parallel patterns)
- **Reading**: Anthropic "Building Effective Agents" (2024)

---

## Questions for Research Phase

Before coding, answer these:

1. **Entity Extraction**: How does LLM distinguish entities vs. attributes?
2. **Graph Construction**: Why is deduplication critical?
3. **Query Answering**: Difference between BFS traversal and vector search?
