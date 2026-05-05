#!/usr/bin/env python3
"""
Main benchmark orchestrator for GraphRAG vs FlatRAG.
Run this after both systems are implemented and tested individually.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from shared import load_corpus, load_benchmark_questions
from graphrag import build_graphrag_system, GraphRAG
from flatrag import FlatRAG
from benchmark import run_benchmark, save_results_csv, generate_report


def main():
    print("\n" + "="*70)
    print("GRAPHRAG vs FLATRAG - FULL BENCHMARK")
    print("="*70)

    # 1. Load corpus
    print("\n[1/5] Loading corpus...")
    documents = load_corpus()
    print(f"  ✓ Loaded {len(documents)} valid documents")

    if len(documents) < 10:
        print("  WARNING: Less than 10 documents. Results may not be meaningful.")

    # 2. Build GraphRAG system
    print("\n[2/5] Building GraphRAG (NetworkX)...")
    print("  This may take a few minutes due to LLM entity extraction...")

    # Use subset for faster testing (remove [:20] for full corpus)
    test_docs = documents[:min(20, len(documents))]
    graphrag = build_graphrag_system(test_docs)

    graph_stats = graphrag.graph.get_stats()
    print(f"  ✓ Graph: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges")

    # 3. Build FlatRAG index
    print("\n[3/5] Building FlatRAG (ChromaDB)...")
    flatrag = FlatRAG()
    flatrag.build_index(test_docs, chunk_size=1000)

    flat_stats = flatrag.get_stats()
    print(f"  ✓ Index: {flat_stats['total_chunks']} chunks")

    # 4. Run benchmark
    print("\n[4/5] Running benchmark on 20 questions...")
    questions = load_benchmark_questions()

    # Wrapper functions for benchmark
    def graphrag_query(q):
        return graphrag.query(q)

    def flatrag_query(q):
        return flatrag.query(q, k=5)

    results = run_benchmark(graphrag_query, flatrag_query, questions)

    # 5. Save results and generate report
    print("\n[5/5] Saving results and generating report...")
    save_results_csv(results)

    report = generate_report(results)

    # Save report
    with open("LAB19_REPORT.md", 'w') as f:
        f.write(report)

    print("\n" + report)
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)

    print("\n📊 Deliverables:")
    print("  1. data/results/benchmark_results.csv - Raw results (fill in 'correct' column)")
    print("  2. LAB19_REPORT.md - Summary report")
    print("  3. data/graph/knowledge_graph.gexf - Graph for visualization")

    print("\n📝 Next steps:")
    print("  1. Open benchmark_results.csv")
    print("  2. For each question, judge if GraphRAG and/or FlatRAG answered correctly")
    print("  3. Fill 'correct' column with 'true' or 'false'")
    print("  4. Re-run: python -c 'from benchmark import generate_report; from benchmark import load_results_csv; print(generate_report(load_results_csv()))'")
    print("  5. Submit LAB19_REPORT.md + CSV + graph visualization")

    print("\n⚠️  IMPORTANT: You must manually judge correctness for each answer!")
    print("   The benchmark runs automatically, but human evaluation is required.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
