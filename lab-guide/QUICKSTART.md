# Quick Start Guide for Team of 2

## 5-Minute Setup (Both Developers)

```bash
# 1. Navigate to project
cd /home/al/Resources/vin/graph-rag-is-dope/lab-guide

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

---

## Developer A: GraphRAG with Neo4j/NetworkX

### Your Tasks:
1. Build knowledge graph using entity extraction
2. Implement BFS traversal
3. Create graph-based retrieval

### Quick Start:
```bash
# Your branch
git checkout -b graphrag-dev

# Test your implementation
python -c "
from src.graphrag import build_graph, graphrag_query
from shared import load_corpus
docs = load_corpus()[:3]
G = build_graph(docs)
answer, latency = graphrag_query('What relationship does Microsoft have with OpenAI?', G)
print(answer)
"
```

### Files to edit:
- `src/graphrag/__init__.py` - main implementation
- (Optional) `src/graphrag/neo4j_builder.py` - if using Neo4j instead of NetworkX

---

## Developer B: Flat RAG with ChromaDB

### Your Tasks:
1. Chunk documents
2. Create embeddings index
3. Implement vector similarity search

### Quick Start:
```bash
# Your branch
git checkout -b flatrag-dev

# Test your implementation
python -c "
from src.flatrag import FlatRAG
from shared import load_corpus
docs = load_corpus()[:3]
rag = FlatRAG()
rag.build_index(docs)
answer, latency = rag.query('What relationship does Microsoft have with OpenAI?')
print(answer)
"
```

### Files to edit:
- `src/flatrag/__init__.py` - main implementation

---

## Coordinated Workflow

### Phase 1: Individual Setup (15 min)
- [ ] Both: clone repo, create venv, install dependencies
- [ ] Both: verify corpus loads correctly
- [ ] Dev A: start NetworkX implementation
- [ ] Dev B: start ChromaDB implementation

### Phase 2: Build Indexes (30 min)
- [ ] Dev A: complete entity extraction → graph building
- [ ] Dev B: complete chunking → vector indexing
- [ ] Both: test with sample query on your system

**Sync Point @ T-75min**: Both systems have working indexes

### Phase 3: Query Implementation (20 min)
- [ ] Dev A: BFS traversal + textualization + LLM generation
- [ ] Dev B: similarity search + LLM generation
- [ ] Both: test with same sample question

**Sync Point @ T-30min**: Both systems can answer questions

### Phase 4: Benchmark (15 min)
- [ ] Both: run `python run_benchmark.py` (or coordinate manually)
- [ ] Check: 20 questions × 2 systems = 40 results in CSV
- [ ] Verify: latency recorded for all queries

### Phase 5: Analysis & Report (20 min)
- [ ] Dev A: write GraphRAG analysis section
- [ ] Dev B: write FlatRAG analysis section
- [ ] Merge into `LAB19_REPORT.md`
- [ ] Review together, finalize

---

## Important Files

| File | Purpose | Who Edits |
|------|---------|-----------|
| `src/graphrag/__init__.py` | GraphRAG implementation | Dev A |
| `src/flatrag/__init__.py` | FlatRAG implementation | Dev B |
| `src/shared/__init__.py` | Shared utilities | Both (careful) |
| `src/shared/benchmark.py` | Benchmark runner | Both |
| `run_benchmark.py` | Main orchestrator | Both (test together) |
| `LAB19_REPORT.md` | Final report | Both (merge) |
| `data/results/benchmark_results.csv` | Raw results | Both (shared) |
| `data/graph/knowledge_graph.gexf` | Graph export | Dev A |

---

## Git Strategy

```bash
# Start
git checkout -b graphrag-dev    # Dev A
git checkout -b flatrag-dev     # Dev B

# Work independently...

# Merge when complete
git checkout main
git merge graphrag-dev
git merge flatrag-dev
git push origin main
```

---

## Troubleshooting

### OpenAI Rate Limits
Add retry logic or slow down:
```python
import time
time.sleep(1)  # Between API calls
```

### ChromaDB Persistence Issues
Delete and rebuild:
```bash
rm -rf data/chroma_db
```

### NetworkX Graph Too Large
Reduce entity extraction or increase LLM context limits.

---

## Communication Checklist

Every 30 minutes, share:
- ✅ What you completed
- ⚠️ Any blockers
- 🎯 What's next
- ❓ What you need from teammate

---

## Lab Deliverables Checklist

- [ ] Source code in `src/`
- [ ] Graph visualization (`data/graph/knowledge_graph.gexf` or screenshot)
- [ ] Benchmark CSV (`data/results/benchmark_results.csv`)
- [ ] Report (`LAB19_REPORT.md`) with:
  - [ ] Accuracy comparison
  - [ ] Latency comparison
  - [ ] 2-3 sample Q&A showing GraphRAG wins
  - [ ] Failure mode analysis
  - [ ] Conclusion (GraphRAG improvement %)

---

## Questions?

Refer to:
- `TEAM_PLAN.md` - detailed 2-hour plan
- `LAB_19_SUMMARY.md` - lab requirements
- `CORPUS_SUMMARY.md` - corpus details

**Good luck! Work in parallel, sync frequently, you'll finish in 2 hours.**
