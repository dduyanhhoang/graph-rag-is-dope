# GraphRAG vs Flat RAG - Lab 19

A comparative implementation of GraphRAG (using Neo4j/NetworkX) and Flat RAG (using ChromaDB) for multi-hop question answering.

## Project Structure

```
.
├── corpus/                    # Curated Wikipedia articles (26+ documents)
│   ├── cleaned_*.txt         # Processed documents
│   └── manifest.json         # Corpus metadata
├── src/
│   ├── graphrag/             # Dev A: GraphRAG implementation
│   │   └── __init__.py       # NetworkX/Neo4j graph building & BFS retrieval
│   ├── flatrag/              # Dev B: Flat RAG implementation
│   │   └── __init__.py       # ChromaDB indexing & vector search
│   └── shared/
│       ├── __init__.py       # Corpus loader, LLM wrapper, benchmark questions
│       └── benchmark.py      # Benchmark runner & result aggregation
├── data/
│   ├── graph/               # Exported graphs (GEXF format)
│   ├── chroma_db/           # ChromaDB persistence
│   └── results/
│       └── benchmark_results.csv  # Raw benchmark results
├── notebooks/               # Jupyter notebooks for analysis
├── scripts/
│   └── curate_corpus.py    # Corpus curation script
├── lab-guide/              # Documentation only
│   ├── LAB_19_SUMMARY.md  # Lab requirements overview
│   ├── TEAM_PLAN.md       # 2-person collaboration plan
│   ├── QUICKSTART.md      # Quick reference
│   ├── CORPUS_SUMMARY.md  # Corpus status
│   └── CORPUS_CURATION.md # Curation guide
├── requirements.txt        # Python dependencies
├── run_benchmark.py       # Main benchmark orchestrator
├── test_components.py     # Individual component tests
└── README.md              # This file
```

---

## Quick Start

### 1. Setup (Both Developers)

```bash
# Clone and navigate
cd /home/al/Resources/vin/graph-rag-is-dope

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 2. Clean Corpus (Important!)

Remove the 4 disambiguation-wrong documents:

```bash
cd corpus
rm cleaned_Amazon_rainforest.txt cleaned_Meta_(prefix).txt \
   cleaned_Nikola_Tesla.txt cleaned_Replicate_(biology).txt
cd ..
```

### 3. Test Individual Components

**Dev A (GraphRAG):**
```bash
python test_components.py --dev A
```

**Dev B (FlatRAG):**
```bash
python test_components.py --dev B
```

### 4. Run Full Benchmark

Both developers should coordinate and run together:

```bash
python run_benchmark.py
```

---

## Team Workflow

| Developer | Stack | Branch | Main File |
|-----------|-------|--------|-----------|
| Dev A | NetworkX/Neo4j | `graphrag-dev` | `src/graphrag/__init__.py` |
| Dev B | ChromaDB | `flatrag-dev` | `src/flatrag/__init__.py` |

### 2-Hour Timeline

| Phase | Duration | Dev A | Dev B |
|-------|----------|-------|-------|
| Setup | 15 min | venv, deps | venv, deps |
| Indexing | 45 min | Entity extraction → Graph | Chunking → Embeddings |
| Query | 30 min | BFS traversal | Vector search |
| Benchmark | 15 min | Run 20 questions | Run 20 questions |
| Report | 15 min | Write analysis | Write analysis |

**Sync every 30 minutes!**

---

## Expected Results

With the curated tech company corpus, you should see:

- **GraphRAG**: 20-40% better accuracy on multi-hop questions
- **Flat RAG**: Faster queries, good for single-document facts
- **Key win**: Questions like "Which companies invested in both OpenAI and Anthropic?"

### Sample Multi-Hop Questions

1. Which AI companies were co-founded by former Google employees?
2. What cloud providers do OpenAI and Anthropic use?
3. Which companies invested in both OpenAI and Anthropic?
4. What relationship does Microsoft have with OpenAI?
5. Who are the main investors in Anthropic?

---

## Lab Deliverables

- [x] Source code in `src/`
- [ ] Graph visualization (`data/graph/knowledge_graph.gexf` or screenshot)
- [ ] Benchmark CSV (`data/results/benchmark_results.csv`)
- [ ] Report (`LAB19_REPORT.md`) with:
  - Accuracy comparison table
  - Latency comparison
  - 2-3 sample Q&A showing GraphRAG wins
  - Failure mode analysis
  - Conclusion with improvement percentage

---

## Documentation

- **`lab-guide/LAB_19_SUMMARY.md`** - Complete lab requirements and theory
- **`lab-guide/TEAM_PLAN.md`** - Detailed 2-hour collaboration plan
- **`lab-guide/QUICKSTART.md`** - Quick reference during development
- **`lab-guide/CORPUS_SUMMARY.md`** - Corpus curation status

---

## Troubleshooting

### OpenAI Rate Limits
Add `time.sleep(1)` between API calls or use caching.

### ChromaDB Issues
Delete and rebuild: `rm -rf data/chroma_db`

### Graph Too Small
Ensure you're extracting entities properly. Check the extracted triples count.

---

## Key Insight

**GraphRAG excels at multi-hop reasoning** because it traverses relationships:
```
(Microsoft)-[INVESTED_IN]->(OpenAI)
                     -> links to
(OpenAI)-[USES]->(Azure)
```

Flat RAG would need to find documents mentioning both Microsoft and OpenAI in the same chunk, which is less likely across a large corpus.

---

**Good luck! Work in parallel, sync frequently, you'll finish in 2 hours.**
