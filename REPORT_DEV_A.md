# GraphRAG Implementation Report

**Developer**: Hoang Dinh Duy Anh (SID: 2A202600064)  
**GitHub**: dduyanhhoang  
**Role**: Dev A - GraphRAG System

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [GraphRAG Implementation](#graphrag-implementation)
3. [Issues Encountered and Fixed](#issues-encountered-and-fixed)
4. [Test Results](#test-results)
5. [Graph Statistics](#graph-statistics)

---

## Executive Summary

This report documents the implementation of GraphRAG (Graph-based Retrieval-Augmented Generation) using NetworkX for the knowledge graph and BFS traversal for multi-hop retrieval.

**Final GraphRAG Accuracy**: 20% (4/20 questions)  
**Average Latency**: 1925ms

The primary bottleneck was entity extraction from user questions, which failed to correctly identify entities due to JSON parsing issues. The knowledge graph itself was structurally sound with 640 nodes and 684 edges, but query execution couldn't reliably find relevant starting points.

---

## GraphRAG Implementation

### File: `src/graphrag/__init__.py`

#### 1. Triple Extraction

Function: `extract_triples_from_document(doc)`

Extracts (subject, predicate, object) triples from document content using LLM.

**Predicates supported**:
- `founded_by` - person founded company
- `co-founded` - person co-founded company
- `developed` - company/product developed something
- `created` - person created product
- `invested_in` - entity invested in company
- `partnered_with` - company partnered with company
- `acquired` - company acquired company
- `uses` - company uses product/service
- `works_at` - person works at company
- `CEO_of` - person is CEO of company
- `based_in` - company based in location

**Prompt design**: Constrained to 30 triples maximum per document, focused on AI/tech companies, people, products, and technologies.

#### 2. KnowledgeGraph Class

```python
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.embedder = UniversalEmbedder()
        self.node_embeddings = {}
```

**Methods**:
- `add_triple(subject, predicate, obj, source_doc)` - adds nodes and edge with metadata
- `build_from_documents(documents)` - iterates through docs, extracts triples
- `compute_node_embeddings()` - generates embeddings for all nodes using UniversalEmbedder
- `export_to_gexf(filename)` - exports to GEXF format for visualization
- `export_to_neo4j(uri, username, password)` - optional Neo4j export
- `get_stats()` - returns graph statistics

#### 3. GraphRAG Query Pipeline

Class: `GraphRAG`

**Query flow**:
1. `extract_entities_from_question(question)` - LLM extracts entity names from question
2. `find_seed_nodes(entities, graph)` - fuzzy matching to find graph nodes
3. `bfs_traverse(graph, seed_nodes, depth=2, max_edges=50)` - breadth-first traversal
4. `textualize_subgraph(triples)` - converts triples to natural language
5. `llm_generate(prompt)` - generates final answer

**Parameters**:
- `traversal_depth`: 2 (default)
- `max_edges`: 50 (default)

#### 4. Helper Functions

- `find_seed_nodes()`: Fuzzy matching with normalization and substring search
- `bfs_traverse()`: BFS with both outgoing and incoming edges for context
- `normalize_entity_name()`: Lowercase, remove spaces/dashes/underscores/dots
- `textualize_subgraph()`: Formats triples as numbered list

---

## Issues Encountered and Fixed

### Issue 1: Missing Imports
**Error**: `NameError: name 'logger' is not defined`  
**Cause**: Used `logger` from shared but didn't import it  
**Fix**: Added to imports:
```python
from shared import (
    llm_generate,
    llm_generate_json,
    load_corpus,
    load_benchmark_questions,
    save_results_csv,
    logger,  # Added
    extract_entities_from_question,
    normalize_entity_name,
    fuzzy_match_entity
)
```

### Issue 2: Missing Path Import
**Error**: `NameError: name 'Path' is not defined`  
**Fix**: Added `from pathlib import Path` at top of file

### Issue 3: GEXF Export Failed
**Error**: `ValueError: too many values to unpack (expected 3)`  
**Cause**: NetworkX GEXF writer couldn't serialize set objects in node attributes  
**Fix**: Convert sets to semicolon-separated strings:
```python
for node, attrs in self.graph.nodes(data=True):
    if 'sources' in attrs and isinstance(attrs['sources'], set):
        attrs['sources'] = ';'.join(sorted(attrs['sources']))
```

### Issue 4: Missing Latency in Early Returns
**Error**: `KeyError: 'latency_ms'`  
**Cause**: Early return paths didn't set latency in metadata  
**Fix**: Added `metadata["latency_ms"] = int((time.time() - start_time) * 1000)` before each return

### Issue 5: Entity Extraction Failure
**Symptom**: LLM returned ````json ["AI", "Google"]```` causing JSON parse failure  
**Root**: `llm_generate_json()` only searched for `{.*}` pattern, missed arrays  
**Fix Location**: `src/shared/__init__.py` (shared module)  
**Fix**: Enhanced to also search for `\[.*\]` pattern:
```python
# Try array pattern first
json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
if json_match:
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
# Try object pattern
json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
...
```

---

## Test Results

### Component Test: `test_components.py --dev A`

```
TESTING GRAPHRAG (Dev A)
==================================================

1. Loading corpus...
   Using 3 documents for test

2. Building knowledge graph...
   Graph: 49 nodes, 52 edges

3. Computing node embeddings...
   Computed embeddings for 49 nodes

4. Exporting graph...
   Exported to data/graph/test_graph.gexf

5. Testing query...
   Q: What relationship does Microsoft have with OpenAI?
   A: I couldn't find information about What, Microsoft in the knowledge graph....
   (Triples: 0, Latency: 946ms)

   Q: Who founded OpenAI?
   A: I couldn't find information about Who in the knowledge graph....
   (Triples: 0, Latency: 813ms)

GraphRAG component test PASSED
```

**Note**: Entity extraction failed on test questions, returning entities like "What", "Who" instead of actual entity names. This is the root cause of poor benchmark performance.

---

## Graph Statistics

### Full Corpus Graph (26 documents)

Built with all valid documents after running `build_graphrag_system(load_corpus())`:

- **Nodes**: 640
- **Edges**: 684
- **Density**: 0.00167
- **Average Degree**: 2.14
- **Connected Components**: 4

### Top 10 Entities by Degree

| Entity | Degree |
|--------|--------|
| Google Cloud Platform | 66 |
| Google | 41 |
| Microsoft | 39 |
| Mistral AI | 34 |
| Microsoft Azure | 33 |
| Salesforce | 32 |
| IBM | 32 |
| Intel Corporation | 32 |
| Amazon Web Services | 31 |
| DeepMind | 31 |

### Export Location

`data/graph/knowledge_graph.gexf` (292KB) - ready for import into Neo4j or Gephi for visualization.

---

## Code Summary

| File | Lines | Purpose |
|------|-------|---------|
| `src/graphrag/__init__.py` | 273 | Complete GraphRAG implementation |
| `src/shared/__init__.py` | 359 | Shared utilities (contributed to) |
| `src/shared/embedder.py` | 165 | UniversalEmbedder (shared) |
| `src/shared/benchmark.py` | 229 | Benchmark framework (shared) |

**Total Dev A contribution**: ~450 lines (GraphRAG + shared benchmark utilities)

---

## Next Steps for GraphRAG Improvement

1. **Fix entity extraction in question parsing** - highest priority
2. **Use embedding similarity** for seed node matching instead of string matching
3. **Filter generic entities** like "AI", "companies", "products" from extraction
4. **Improve triple extraction** - investigate why some docs yield 0 triples
5. **Merge duplicate nodes** - e.g., "Google" and "Google Cloud Platform" should be consolidated

---
