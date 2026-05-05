# GraphRAG vs FlatRAG - Group Report

**Team Members**:
- **Hoang Dinh Duy Anh** (SID: 2A202600064, GitHub: dduyanhhoang) - GraphRAG Implementation
- **Tran Nhat Vi** (SID: 2A202600497, GitHub: trannhatvi-ai) - FlatRAG Implementation

**Course**: Lab 19 - GraphRAG vs FlatRAG  
**Date**: May 5, 2026

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Analysis](#analysis)
6. [Collaboration](#collaboration)
7. [Conclusions](#conclusions)
8. [Deliverables](#deliverables)

---

## Executive Summary

This project compared two RAG (Retrieval-Augmented Generation) architectures on a corpus of Wikipedia articles about AI/tech companies:

- **GraphRAG** (Hoang): Knowledge graph with BFS traversal for multi-hop reasoning
- **FlatRAG** (Vi): Vector similarity search using ChromaDB

**Key Finding**: FlatRAG achieved 50% accuracy vs GraphRAG's 20%, contrary to expectations. The primary cause was entity extraction failures in GraphRAG's query pipeline.

**Graph Built**: 640 nodes, 684 edges from 26 documents  
**Benchmark**: 20 questions, both systems fully evaluated  
**Deliverables**: Code, data, visualizations, and comprehensive reports

---

## Project Overview

### Objective
Compare GraphRAG (graph-based multi-hop retrieval) vs FlatRAG (vector-based retrieval) on a corpus of tech company Wikipedia articles.

### Hypothesis
GraphRAG would outperform FlatRAG on multi-hop questions requiring relationships between entities.

### Corpus
- **Source**: Wikipedia articles about AI/tech companies
- **Size**: 26 curated documents (after removing 4 disambiguation/wrong files)
- **Companies**: OpenAI, Anthropic, Microsoft, Google, Amazon, NVIDIA, Cohere, DeepMind, Intel, IBM, and others
- **Content Types**: Founding information, investments, partnerships, products, cloud providers, acquisitions

### Team Structure
- **Hoang**: GraphRAG implementation using NetworkX
- **Vi**: FlatRAG implementation using ChromaDB
- **Shared**: Both used `src/shared/` for common utilities (corpus loader, embedder, benchmark)

---

## Methodology

### 1. Corpus Preparation

**Source**: `corpus/` directory with `cleaned_*.txt` files

**Filtering**: Removed 4 incorrect documents:
- `cleaned_Amazon_rainforest.txt`
- `cleaned_Meta_(prefix).txt`
- `cleaned_Nikola_Tesla.txt`
- `cleaned_Replicate_(biology).txt`

**Loading**: `load_corpus()` in `src/shared/__init__.py`
- Skips filtered files automatically
- Returns list of dicts: `{doc_id, title, content, filepath}`

**Result**: 26 valid documents used

### 2. GraphRAG System (Hoang)

**Implementation**: `src/graphrag/__init__.py`

**Pipeline**:
1. **Triple Extraction** - LLM extracts (subject, predicate, object) from each document
2. **Graph Building** - NetworkX DiGraph stores entities as nodes, relationships as edges
3. **Embedding** - UniversalEmbedder creates embeddings for all nodes
4. **Query**:
   - Extract entities from question (LLM)
   - Find seed nodes via fuzzy matching
   - BFS traversal (depth=2, max_edges=50)
   - Textualize subgraph
   - Generate answer

**Graph Stats**:
- 640 nodes
- 684 edges
- Density: 0.00167
- Connected components: 4

### 3. FlatRAG System (Vi)

**Implementation**: `src/flatrag/__init__.py`

**Pipeline**:
1. **Chunking** - RecursiveCharacterTextSplitter (size=1000, overlap=200)
2. **Indexing** - ChromaDB with UniversalEmbedder
3. **Query**:
   - Embed query
   - Similarity search (k=5)
   - Build context from chunks
   - Generate answer

**Index Stats** (20 docs):
- ~800-1000 chunks
- Persistent storage in `data/chroma_db/`

### 4. Shared Components

**`src/shared/embedder.py`** - UniversalEmbedder
- Supports OpenAI embeddings or Sentence Transformers
- Same embedding model used by both systems for fair comparison

**`src/shared/__init__.py`**
- `llm_generate()` - OpenAI API wrapper with retries
- `llm_generate_json()` - JSON parsing with fallback extraction
- `extract_entities_from_question()` - Entity extraction for GraphRAG
- `load_benchmark_questions()` - 20 multi-hop questions
- `save_results_csv()`, `load_results_csv()` - Results management
- `calculate_metrics()`, `generate_report()` - Report generation

### 5. Benchmark Execution

**Script**: `run_benchmark.py`

**Process**:
1. Load corpus (26 docs)
2. Build GraphRAG from 20 docs + export GEXF
3. Build FlatRAG index from same 20 docs
4. Run all 20 questions through both systems
5. Save results to `data/results/benchmark_results.csv`
6. Generate report

**Metrics Collected**:
- Latency per query
- GraphRAG: triples count, seed nodes, entities
- FlatRAG: retrieved chunks, context chars
- Correctness (manually judged)

### 6. Evaluation

**Manual Judgment**: Open `benchmark_results.csv`, fill `correct` column (True/False) for each answer based on corpus content.

**Automated Script**: `evaluate_results.py` - rule-based correctness validator to standardize evaluation.

---

## Results

### Performance Summary

| Metric | GraphRAG | FlatRAG |
|--------|----------|---------|
| **Accuracy** | 20.0% (4/20) | 50.0% (10/20) |
| **Avg Latency** | 1925ms | 1998ms |
| **Triples/Query** | 32 avg | N/A |
| **Chunks Retrieved** | N/A | 5 avg |

### Question-by-Question Results

| Q# | Question | GraphRAG | FlatRAG |
|----|----------|----------|---------|
| 1 | Which AI companies were co-founded by former Google employees? | ✗ | ✓ (Cohere) |
| 2 | What cloud providers do OpenAI and Anthropic use? | ✗ | ✗ |
| 3 | Which companies invested in both OpenAI and Anthropic? | ✗ | ✗ |
| 4 | Who are the founders of OpenAI and what companies did they work at before? | ✗ | ✗ |
| 5 | What relationship does Microsoft have with OpenAI? | ✗ | ✓ |
| 6 | Which companies partnered with both Amazon and Google? | ✗ | ✓ (Anthropic) |
| 7 | What products did Anthropic develop? | ✓ (Claude) | ✓ |
| 8 | Who invests in AI safety research? | ✗ | ✗ |
| 9 | Which tech companies acquired AI startups in 2024-2025? | ✗ | ✓ (Anthropic-Bun) |
| 10 | What is the connection between NVIDIA and major AI labs? | ✓ | ✓ |
| 11 | Which companies use Azure for AI infrastructure? | ✗ | ✗ |
| 12 | Who are the CEOs of OpenAI and Anthropic? | ✗ | ✓ (Dario Amodei) |
| 13 | What funding rounds did OpenAI have in 2025? | ✗ | ✗ |
| 14 | Which companies have partnerships with Palantir? | ✗ | ✗ |
| 15 | What hardware does OpenAI use for training? | ✗ | ✗ |
| 16 | Which AI companies have DoD contracts? | ✗ | ✗ |
| 17 | What is the relationship between DeepMind and Google? | ✓ (subsidiary) | ✓ |
| 18 | Which companies developed large language models? | ✗ | ✓ |
| 19 | Who are the main investors in Anthropic? | ✓ | ✓ |
| 20 | What cloud partnerships does Google have with AI companies? | ✗ | ✗ |

**Legend**: ✓ = Correct, ✗ = Incorrect

### Sample Answers

**Q1: Which AI companies were co-founded by former Google employees?**

- GraphRAG: `I don't have enough information.`
- FlatRAG: `The AI company co-founded by former Google employees is Cohere, which was founded in 2019 by Aidan Gomez and Nick Frosst, both researchers at Google Brain.`

**Q10: What is the connection between NVIDIA and major AI labs?**

- GraphRAG: `NVIDIA has connections with major AI labs through investments and collaborations. Specifically, NVIDIA has invested in AI21 Labs (relationship 2) and has collaborations with Inflection AI, Inc. (relationships 1 and 23). Additionally, Anthropic has in...`
- FlatRAG: `NVIDIA has a significant connection with major AI labs through its investments and partnerships. In November 2025, NVIDIA and Microsoft were expected to invest up to $15 billion in Anthropic, a major AI lab. Additionally, Anthropic announced plans to...`

---

## Analysis

### Why FlatRAG Outperformed GraphRAG

**30 percentage point gap** (FlatRAG +30% over GraphRAG) was primarily due to:

1. **Entity Extraction Failure in GraphRAG**
   - LLM returned JSON with code fences: ````json ["AI", "Google"]````
   - JSON parser failed, fallback extracted only capitalized words
   - Extracted entities: "AI", "Google" instead of "OpenAI", "Anthropic"
   - Seed node lookup couldn't find relevant graph nodes

2. **Fuzzy Matching Issues**
   - Matching algorithm found wrong entities (e.g., "What" from "What relationship...")
   - Normalization insufficient for entity variations
   - Graph contained correct nodes but retrieval failed to match

3. **Graph Quality vs Query Access**
   - Graph was structurally sound (640 nodes, 684 edges)
   - Some documents yielded 0 triples (Adept, Adobe, Apple, Oracle)
   - But even with good graph, query pipeline couldn't access it

4. **FlatRAG's Robust Simplicity**
   - Vector search directly matches query semantics to document chunks
   - No entity extraction bottleneck
   - Retrieved relevant context for most questions

### Multi-Hop Question Performance

Both systems struggled with:
- Synthesis across multiple entities (Q2, Q3, Q11)
- Specific temporal details (Q13)
- DoD contracts (Q16)
- Cloud partnerships (Q20)

GraphRAG showed theoretical advantage on Q10 (NVIDIA connections) where graph contained the relationships and BFS successfully retrieved them. But this was the exception.

### Latency

- GraphRAG: 1925ms (LLM entity extraction + BFS + generation)
- FlatRAG: 1998ms (embedding lookup + generation)
- Difference negligible (< 100ms), not statistically significant

---

## Collaboration

### Work Division

| Developer | System | Primary Files | Test Command |
|-----------|--------|---------------|--------------|
| Hoang | GraphRAG | `src/graphrag/__init__.py` | `python test_components.py --dev A` |
| Vi | FlatRAG | `src/flatrag/__init__.py` | `python test_components.py --dev B` |

### Shared Work

Both developers contributed to:
- `src/shared/benchmark.py` - Benchmark framework
- Bug fixes across all shared modules
- Testing and validation

### Integration Timeline

1. **Day 1**: Individual implementations completed, component tests passing
2. **Day 2**: Integrated systems, fixed import/export issues, ran full benchmark
3. **Day 3**: Evaluated results, generated visualizations, wrote reports

### Communication

- Synced every 30 minutes during development
- Joint debugging of cross-system issues (benchmark schema, GEXF export)
- Collaborative review of evaluation criteria

---

## Conclusions

### Hypothesis Verdict: NOT SUPPORTED

Expected: GraphRAG > FlatRAG on multi-hop questions  
Result: FlatRAG (50%) > GraphRAG (20%) by 30 percentage points

### Root Cause Analysis

GraphRAG's theoretical advantages (multi-hop traversal, explicit relationships) were negated by implementation issues:

1. **Entity extraction broken** - couldn't parse question entities correctly
2. **Seed matching ineffective** - fuzzy string matching too brittle
3. **Graph not accessed** - queries returned "no information" despite graph containing answers

If entity extraction were fixed, GraphRAG might show its theoretical advantage. As implemented, FlatRAG's simpler approach proved more robust.

### When GraphRAG Would Excel

The architecture is theoretically superior for:
- "What path connects Company X to Company Y through investments?"
- "Which investors are connected to both A and B via intermediaries?"
- "Trace the funding chain from Investor X to Company Y"

These require multi-hop traversal that vector search would struggle with if information is scattered.

### Recommendations

**For GraphRAG**:
1. Fix entity extraction (clean JSON output, filter generic entities)
2. Use embedding similarity for node matching instead of string matching
3. Improve triple extraction rate (some docs yielded 0 triples)
4. Merge duplicate nodes ("Google" vs "Google Cloud Platform")

**For FlatRAG**:
1. Experiment with k=10 or k=15 for retrieval
2. Add re-ranking with cross-encoder
3. Try hybrid search (vector + BM25)

**Benchmark Improvements**:
1. Use LLM-as-judge for evaluation consistency
2. Include partial credit for partially correct answers
3. Increase questions to 50+ for statistical significance

---

## Deliverables

### Code Files

| File | Lines | Author | Description |
|------|-------|--------|-------------|
| `src/graphrag/__init__.py` | 273 | Hoang | GraphRAG implementation |
| `src/flatrag/__init__.py` | 217 | Vi | FlatRAG implementation |
| `src/shared/__init__.py` | 359 | Shared | Corpus, LLM, entities, results |
| `src/shared/embedder.py` | 165 | Shared | UniversalEmbedder |
| `src/shared/benchmark.py` | 229 | Shared | Benchmark framework |
| `run_benchmark.py` | ~100 | Shared | Main orchestrator |
| `test_components.py` | ~200 | Shared | Component tests |
| `evaluate_results.py` | ~150 | Shared | Automated evaluation |

**Total**: ~1500 lines of Python code

### Data Files

| File | Size | Description |
|------|------|-------------|
| `data/graph/knowledge_graph.gexf` | 292KB | Full knowledge graph (640 nodes, 684 edges) |
| `data/graph/visualization.png` | 796KB | Static graph visualization (3980x3178) |
| `data/graph/interactive.html` | 317KB | Interactive browser visualization |
| `data/results/benchmark_results.csv` | 12KB | All 40 results with correctness judgments |

### Documentation

| File | Purpose |
|------|---------|
| `REPORT_DEV_A.md` | Hoang's individual implementation report |
| `REPORT_DEV_B.md` | Vi's individual implementation report |
| `LAB19_REPORT.md` | This group report (this file) |
| `README.md` | Project setup and quick start |
| `lab-guide/LAB_19_SUMMARY.md` | Original lab requirements |
| `lab-guide/TEAM_PLAN.md` | Team collaboration plan |
| `CORPUS_CURATION.md` | Corpus curation guide |
| `IMPLEMENTATION_COMPLETE.md` | Implementation overview |

---

## Appendix: Full Results Table

| Q | System | Answer Summary | Correct |
|---|--------|----------------|---------|
| 1 | GraphRAG | "I don't have enough information" | ✗ |
| 1 | FlatRAG | "Cohere founded by former Google Brain researchers" | ✓ |
| 2 | GraphRAG | "OpenAI not specified, Anthropic uses AWS" | ✗ |
| 2 | FlatRAG | "OpenAI uses Azure, no info on Anthropic" | ✗ |
| 3 | GraphRAG | "I don't have enough information" | ✗ |
| 3 | FlatRAG | "I don't have enough information" | ✗ |
| 4 | GraphRAG | "I don't have enough information" | ✗ |
| 4 | FlatRAG | "I don't have enough information" | ✗ |
| 5 | GraphRAG | "I don't have enough information" | ✗ |
| 5 | FlatRAG | "Microsoft invested in OpenAI, partnership" | ✓ |
| 6 | GraphRAG | "I don't have enough information" | ✗ |
| 6 | FlatRAG | "Anthropic partnered with both" | ✓ |
| 7 | GraphRAG | "Anthropic developed Claude models" | ✓ |
| 7 | FlatRAG | "Anthropic developed Claude" | ✓ |
| 8 | GraphRAG | "I don't have enough information" | ✗ |
| 8 | FlatRAG | "I don't have enough information" | ✗ |
| 9 | GraphRAG | "I don't have enough information" | ✗ |
| 9 | FlatRAG | "Anthropic acquired Bun in Dec 2025" | ✓ |
| 10 | GraphRAG | "NVIDIA invested in AI21 Labs, Inflection" | ✓ |
| 10 | FlatRAG | "NVIDIA/Microsoft invested in Anthropic" | ✓ |
| 11 | GraphRAG | "I don't have enough information" | ✗ |
| 11 | FlatRAG | "I don't have enough information" | ✗ |
| 12 | GraphRAG | "I don't have enough information" | ✗ |
| 12 | FlatRAG | "Dario Amodei is CEO of Anthropic" | ✓ |
| 13 | GraphRAG | "I don't have enough information" | ✗ |
| 13 | FlatRAG | "I don't have enough information" | ✗ |
| 14 | GraphRAG | "I don't have enough information" | ✗ |
| 14 | FlatRAG | "I don't have enough information" | ✗ |
| 15 | GraphRAG | "I don't have enough information" | ✗ |
| 15 | FlatRAG | "I don't have enough information" | ✗ |
| 16 | GraphRAG | "I don't have enough information" | ✗ |
| 16 | FlatRAG | "I don't have enough information" | ✗ |
| 17 | GraphRAG | "DeepMind is a subsidiary of Google" | ✓ |
| 17 | FlatRAG | "DeepMind is Google's AI lab" | ✓ |
| 18 | GraphRAG | "I don't have enough information" | ✗ |
| 18 | FlatRAG | "OpenAI, Anthropic, Google developed LLMs" | ✓ |
| 19 | GraphRAG | "Amazon, Google, Lightspeed, others invested" | ✓ |
| 19 | FlatRAG | "Amazon and Google invested in Anthropic" | ✓ |
| 20 | GraphRAG | "I don't have enough information" | ✗ |
| 20 | FlatRAG | "I don't have enough information" | ✗ |

---

