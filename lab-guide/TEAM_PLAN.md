# Team Collaboration Plan: GraphRAG Lab (2 Developers)

## Overview

**Goal**: Build and compare Flat RAG (ChromaDB) vs GraphRAG (Neo4j) systems
**Team Size**: 2 developers
**Timebox**: 2 hours
**Strategy**: Parallel development with integration at milestones

---

## Role Split

| Developer | Stack | Primary Tasks |
|-----------|-------|---------------|
| **Dev A** (You) | Neo4j + NetworkX | Corpus loading → Entity extraction → Graph building → GraphRAG retrieval |
| **Dev B** (Teammate) | ChromaDB + LangChain | Corpus loading → Chunking → Embeddings → Flat RAG retrieval |

**Why this split**: Both need to start from the same corpus, then diverge on different backends. Allows parallel work with minimal merge conflicts.

---

## Shared Prerequisites (Both Developers)

### 1. Environment Setup (15 min sync)

Create a shared `requirements.txt`:

```txt
# Common dependencies
openai>=1.0.0
python-dotenv>=1.0.0
pandas>=2.0.0
tqdm>=4.0.0
langchain>=0.1.0
langchain-text-splitters>=0.0.1

# Dev A (GraphRAG)
networkx>=3.0
neo4j>=5.0.0
matplotlib>=3.0.0

# Dev B (Flat RAG)
chromadb>=0.4.0
sentence-transformers>=2.0.0
```

Both should:
- Install from same `requirements.txt`
- Use same OpenAI API key (set in `.env`)
- Access same `corpus/` directory
- Clone/fork the repo to separate branches: `graphrag-dev` and `flatrag-dev`

---

## 2-Hour Timeline (Parallel Tracks)

### **PHASE 1: Corpus Loading & Preprocessing (30 min)**

**Dev A (GraphRAG)** | **Dev B (Flat RAG)** | **Integration Point**
---------------------|---------------------|--------------------
Load cleaned corpus files | Load cleaned corpus files | Same file list
Minimal chunking (keep docs whole or large chunks) | Smart chunking (~500-1000 tokens) | Share chunking strategy
Output: List of `(doc_id, title, content)` | Output: List of `(chunk_id, doc_id, content)` | Both start from same docs

**Deliverable**: Both have loaded corpus in memory, ready for next step

---

### **PHASE 2: Indexing (45 min)**

**Dev A (GraphRAG)** | **Dev B (Flat RAG)** | **Integration Point**
---------------------|---------------------|--------------------
**Option 1 (Fast)**: LLM extraction → triples → NetworkX | Generate embeddings for chunks → ChromaDB | **Share entity list** (Dev A sends entities found, Dev B can enhance search)
**Option 2 (Production)**: LLM extraction → triples → Neo4j | | Both systems ready to query by T-75min

**Dev A Detail**:
```python
# Entity extraction prompt (run on each document)
PROMPT = """
Extract entities and relationships from this text.

Output format (one triple per line):
(subject)|(predicate)|(object)

Examples:
(Sam Altman)|(co-founded)|(OpenAI)
(OpenAI)|(developed)|(GPT-4)
(OpenAI)|(uses)|(Microsoft Azure)

Text: {text}
"""

# Build NetworkX graph
import networkx as nx
G = nx.DiGraph()
for subj, pred, obj in triples:
    G.add_edge(subj, obj, relation=pred, source_doc=doc_id)
```

**Dev B Detail**:
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="flatrag",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add chunks with metadata
for chunk in chunks:
    vectorstore.add_texts(
        texts=[chunk],
        metadatas=[{"doc_id": doc_id, "title": title}]
    )
```

**Sync Point @ T-45min**: Quick check-in - both have working index. Dev A shares initial entity list.

---

### **PHASE 3: Query Implementation (30 min)**

**Dev A (GraphRAG)** | **Dev B (Flat RAG)** | **Integration Point**
---------------------|---------------------|--------------------
Implement BFS traversal | Implement similarity search | **Same query set** (20 multi-hop questions)
Textualization step | RAG retrieval + LLM | Share benchmark questions
LLM generation prompt | LLM generation prompt | **Same evaluation criteria**

**Dev A - GraphRAG Query**:
```python
def graphrag_query(question: str, G: nx.DiGraph, depth: int = 2):
    # 1. Extract entities from question (use LLM or simple NER)
    entities = llm_extract_entities(question)

    # 2. Find seed nodes (fuzzy match)
    seed_nodes = []
    for entity in entities:
        matches = [n for n in G.nodes() if entity.lower() in n.lower()]
        seed_nodes.extend(matches[:3])  # Top 3 matches

    # 3. BFS traversal
    subgraph = []
    for seed in seed_nodes:
        # Get neighbors up to depth
        for node, attrs in G.nodes(data=True):
            # BFS logic here
            pass
        # Collect edges
        for u, v, data in G.edges(seed, data=True):
            subgraph.append(f"({u})-[{data['relation']}]->({v})")

    # 4. Textualization
    context = "\n".join(subgraph[:50])  # Limit to context window

    # 5. Generate answer
    prompt = f"""
    Question: {question}
    Context from knowledge graph:
    {context}

    Answer based ONLY on the provided relationships:
    """
    return llm_generate(prompt)
```

**Dev B - Flat RAG Query**:
```python
def flatrag_query(question: str, vectorstore):
    # 1. Similarity search
    results = vectorstore.similarity_search(
        question, k=5, filter=None
    )

    # 2. Build context
    context = "\n\n".join([r.page_content for r in results])

    # 3. Generate answer
    prompt = f"""
    Question: {question}
    Context from documents:
    {context}

    Answer based ONLY on the provided context:
    """
    return llm_generate(prompt)
```

**Sync Point @ T-15min**: Both query implementations ready. Share the 20 benchmark questions.

---

### **PHASE 4: Benchmarking (15 min)**

**Both Developers** run the **same 20 multi-hop questions**:

1. "Which AI companies were co-founded by former Google employees?"
2. "What cloud providers do OpenAI and Anthropic use?"
3. "Which companies invested in both OpenAI and Anthropic?"
4. "Who are the founders of OpenAI and what companies did they work at before?"
5. "What relationship does Microsoft have with OpenAI?"
6. "Which companies partnered with both Amazon and Google?"
7. "What products did Anthropic develop?"
8. "Who invests in AI safety research?"
9. "Which tech companies acquired AI startups in 2024-2025?"
10. "What is the connection between NVIDIA and major AI labs?"
11. "Which companies use Azure for AI infrastructure?"
12. "Who are the CEOs of OpenAI and Anthropic?"
13. "What funding rounds did OpenAI have in 2025?"
14. "Which companies have partnerships with Palantir?"
15. "What hardware does OpenAI use for training?"
16. "Which AI companies have DoD contracts?"
17. "What is the relationship between DeepMind and Google?"
18. "Which companies developed large language models?"
19. "Who are the main investors in Anthropic?"
20. "What cloud partnerships does Google have with AI companies?"

**Recording Template** (create `benchmark_results.csv`):

```csv
question_id,question,system,answer,correct,latency_ms,tokens_used,notes
1,"Which AI companies were co-founded by former Google employees?",graphrag,"Anthropic was co-founded by Dario Amodei...",True,1250,450,"Got all 3 companies"
1,"Which AI companies were co-founded by former Google employees?",flatrag,"I don't have enough information...",False,850,280,"Hallucinated - said Google"
```

---

### **PHASE 5: Analysis & Report (15 min)**

**Dev A** (GraphRAG) | **Dev B** (Flat RAG)** | **Combined**
---------------------|----------------------|------------
Calculate accuracy % | Calculate accuracy % | Compare both systems
Count hallucination cases | Count hallucination cases | Identify failure patterns
Avg latency per query | Avg latency per cost | Cost analysis (tokens)
Write GraphRAG section | Write Flat RAG section | Merge reports

**Report Structure** (`LAB19_REPORT.md`):

```markdown
# Lab 19: GraphRAG vs Flat RAG Benchmark Report

## 1. Setup
- Corpus: 26 documents, ~700K words
- GraphRAG: Neo4j + NetworkX, X triples, Y nodes
- Flat RAG: ChromaDB + sentence-transformers

## 2. Results

### Accuracy Comparison
| Metric | GraphRAG | Flat RAG |
|--------|----------|----------|
| Multi-hop accuracy | XX% | XX% |
| Hallucination rate | X% | X% |
| Factoid accuracy | XX% | XX% |

### Performance
| Metric | GraphRAG | Flat RAG |
|--------|----------|----------|
| Avg latency | Xms | Xms |
| Tokens per query | XXX | XXX |
| Index build time | Xmin | Xmin |

## 3. Analysis
- GraphRAG outperformed by X% on multi-hop questions
- Flat RAG faster but less accurate on relational queries
- Key failure modes: [list]
- Cost comparison: GraphRAG indexing expensive, queries cheap

## 4. Sample Outputs
[Show 2-3 examples where GraphRAG succeeded and Flat RAG failed]

## 5. Conclusion
GraphRAG provides [X]% improvement in multi-hop accuracy, validating the hypothesis.
```

---

## Communication Strategy

### Check-in Points (standup style):
- **T-90min**: Both confirm corpus loaded successfully
- **T-45min**: Both confirm indexes built. Share entity count.
- **T-15min**: Both have query implementations. Agree on benchmark questions.
- **T-5min**: Both finished benchmarking. Share raw results.
- **T+15min**: Combine reports, final review

### Tools for Collaboration:
- **Shared Google Doc** or **Notion page** for real-time notes
- **Git branches**: `graphrag-dev` and `flatrag-dev`, merge into `lab19-combined`
- **Discord/Slack** for quick questions
- **Screen share** if stuck on bugs

---

## File Structure

```
lab-guide/
├── corpus/                          # Shared corpus (already curated)
├── src/
│   ├── graphrag/                   # Dev A's code
│   │   ├── __init__.py
│   │   ├── extractor.py            # Entity/relation extraction
│   │   ├── graph_builder.py        # NetworkX/Neo4j builder
│   │   ├── retriever.py            # BFS traversal
│   │   └── generator.py            # LLM answer generation
│   ├── flatrag/                    # Dev B's code
│   │   ├── __init__.py
│   │   ├── indexer.py              # Chunking + embeddings
│   │   ├── retriever.py            # ChromaDB search
│   │   └── generator.py            # LLM answer generation
│   └── shared/
│       ├── llm.py                  # Shared LLM wrapper
│       ├── corpus_loader.py        # Shared corpus loading
│       └── benchmark.py            # Shared benchmark runner
├── notebooks/
│   ├── 01_graph_exploration.ipynb  # Dev A: Graph visualization
│   ├── 02_flatrag_analysis.ipynb  # Dev B: Vector analysis
│   └── 03_comparison.ipynb        # Combined comparison charts
├── data/
│   ├── graph/                     # Graph data exports
│   ├── chroma_db/                 # Chroma persistence
│   └── results/
│       └── benchmark_results.csv  # Combined results
├── LAB19_REPORT.md                # Final combined report
└── requirements.txt
```

---

## Integration Points & Dependencies

### 1. **Corpus** (Shared)
Both read from `corpus/cleaned_*.txt`. No modification.

### 2. **Entity List** (Dev A → Dev B)
After Phase 1, Dev A shares extracted entities. Dev B can optionally weight these in search.

### 3. **Benchmark Questions** (Both agree)
Both use same 20 multi-hop questions. Dev B creates the master list.

### 4. **Results Aggregation** (Dev B → Dev A)
Dev B collects both CSVs, creates comparison charts. Dev A reviews.

### 5. **Report Merging** (Both → Combined)
Dev A writes GraphRAG section, Dev B writes Flat RAG section, Dev A merges.

---

## Risk Mitigation

| Risk | Prevention | Contingency |
|------|-------------|-------------|
| **Network issues** (API rate limits) | Cache LLM responses locally | Reduce benchmark to 10 questions |
| **Neo4j not installed** | Use NetworkX (no install) | Fallback to NetworkX-only |
| **ChromaDB issues** | Have Chroma in-memory fallback | Use simple FAISS instead |
| **Inconsistent results** | Use same LLM model/temperature | Document differences in report |
| **Time overrun** | Prioritize: indexing > queries > report | Skip visualizations, do minimal report |

---

## Success Criteria

By the end of 2 hours, you should have:

- [ ] **Both systems running** (GraphRAG + Flat RAG)
- [ ] **Same corpus** (26+ documents loaded)
- [ ] **Benchmark completed** on 20 multi-hop questions
- [ ] **Results CSV** with accuracy, latency, token counts
- [ ] **Comparison report** with at least:
  - Accuracy comparison table
  - 2-3 sample outputs showing GraphRAG win
  - Analysis of failure modes
  - Conclusion on GraphRAG value

---

## Quick Start Commands

**Both developers run:**
```bash
cd /home/al/Resources/vin/graph-rag-is-dope/lab-guide

# Create virtual env
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up OpenAI key
echo "OPENAI_API_KEY=your-key-here" > .env
```

**Dev A:**
```bash
# Start Neo4j (if using)
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j

# Work in graphrag branch
git checkout -b graphrag-dev
# Implement src/graphrag/*.py
```

**Dev B:**
```bash
# Work in flatrag branch
git checkout -b flatrag-dev
# Implement src/flatrag/*.py
```

---

## Communication Template

**Daily standup messages** (every 30 min):

```
[Dev A] Status: [what done]
Blockers: [any issues]
Next: [what's next]
Need from Dev B: [specific ask]

[Dev B] Status: [what done]
Blockers: [any issues]
Next: [what's next]
Need from Dev A: [specific ask]
```

---

## After Lab: Merge & Polish

1. Both push branches to GitHub
2. Create PR to `main` with combined report
3. One person merges after review
4. Clean up temporary branches

---

## Appendix: Sample Benchmark Questions

See the 20 questions in the main plan. Both developers must agree on the exact wording before starting Phase 4.

---

