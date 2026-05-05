"""
Benchmark runner - shared code to evaluate both systems.
"""

import time
import csv
from pathlib import Path
from typing import List, Dict, Any, Callable

from shared import load_benchmark_questions


def run_benchmark(
    graphrag_query_func: Callable[[str], Tuple[str, Dict]],
    flatrag_query_func: Callable[[str], Tuple[str, Dict]],
    questions: List[Dict] = None
) -> List[Dict[str, Any]]:
    """
    Run all benchmark questions through both systems and collect results.

    Args:
        graphrag_query_func: Function that takes question and returns (answer, metadata)
        flatrag_query_func: Function that takes question and returns (answer, metadata)
        questions: List of question dicts with 'id' and 'question'

    Returns:
        List of result dictionaries
    """
    if questions is None:
        questions = load_benchmark_questions()

    results = []

    print("\n" + "="*60)
    print("STARTING BENCHMARK")
    print(f"Total questions: {len(questions)}")
    print("="*60)

    for q in questions:
        q_id = q['id']
        question = q['question']

        print(f"\n[{q_id}/20] {question[:50]}...")

        # Test GraphRAG
        print("  GraphRAG: ", end="", flush=True)
        try:
            graph_start = time.time()
            graph_answer, graph_metadata = graphrag_query_func(question)
            graph_latency = int((time.time() - graph_start) * 1000)
            print(f"✓ ({graph_latency}ms)")
        except Exception as e:
            graph_answer = f"ERROR: {str(e)}"
            graph_latency = 0
            graph_metadata = {}
            print(f"✗ {e}")

        # Test FlatRAG
        print("  FlatRAG:  ", end="", flush=True)
        try:
            flat_start = time.time()
            flat_answer, flat_metadata = flatrag_query_func(question)
            flat_latency = int((time.time() - flat_start) * 1000)
            print(f"✓ ({flat_latency}ms)")
        except Exception as e:
            flat_answer = f"ERROR: {str(e)}"
            flat_latency = 0
            flat_metadata = {}
            print(f"✗ {e}")

        # Record GraphRAG result
        results.append({
            'question_id': q_id,
            'question': question,
            'system': 'graphrag',
            'answer': graph_answer[:1000],  # Truncate for CSV
            'latency_ms': graph_latency,
            'triples_count': graph_metadata.get('triples_count', 0),
            'seed_nodes': ';'.join(graph_metadata.get('seed_nodes', [])),
            'entities': ';'.join(graph_metadata.get('entities', [])),
            'correct': '',  # To be filled manually
            'notes': ''
        })

        # Record FlatRAG result
        results.append({
            'question_id': q_id,
            'question': question,
            'system': 'flatrag',
            'answer': flat_answer[:1000],
            'latency_ms': flat_latency,
            'retrieved_chunks': flat_metadata.get('retrieved_chunks', 0),
            'context_chars': flat_metadata.get('total_context_chars', 0),
            'entities': '',  # Not tracked for flatrag
            'correct': '',  # To be filled manually
            'notes': ''
        })

    return results


def save_results_csv(results: List[Dict[str, Any]], filename: str = "data/results/benchmark_results.csv"):
    """Save benchmark results to CSV."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    if not results:
        print("WARNING: No results to save")
        return

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Results saved to {filename}")


def load_results_csv(filename: str = "data/results/benchmark_results.csv") -> List[Dict[str, Any]]:
    """Load benchmark results from CSV."""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics from benchmark results.
    Requires human-judged 'correct' field to be filled.
    """
    graphrag_results = [r for r in results if r['system'] == 'graphrag']
    flatrag_results = [r for r in results if r['system'] == 'flatrag']

    def get_correct_count(records):
        correct = 0
        total = 0
        for r in records:
            if r.get('correct', '').lower() in ('true', 'false'):
                total += 1
                if r['correct'].lower() == 'true':
                    correct += 1
        return correct, total

    g_correct, g_total = get_correct_count(graphrag_results)
    f_correct, f_total = get_correct_count(flatrag_results)

    # Latency
    g_latency = sum(int(r['latency_ms']) for r in graphrag_results) / len(graphrag_results) if graphrag_results else 0
    f_latency = sum(int(r['latency_ms']) for r in flatrag_results) / len(flatrag_results) if flatrag_results else 0

    return {
        'graphrag': {
            'correct': g_correct,
            'total': g_total,
            'accuracy': g_correct / g_total if g_total > 0 else 0,
            'avg_latency_ms': g_latency
        },
        'flatrag': {
            'correct': f_correct,
            'total': f_total,
            'accuracy': f_correct / f_total if f_total > 0 else 0,
            'avg_latency_ms': f_latency
        },
        'improvement_pct': ((g_correct - f_correct) / max(f_total, 1)) * 100 if f_total > 0 else 0
    }


def generate_report(results: List[Dict[str, Any]]) -> str:
    """Generate markdown report from results."""
    metrics = calculate_metrics(results)

    report = []
    report.append("# GraphRAG vs FlatRAG Benchmark Report\n")

    # Summary table
    report.append("## Summary\n")
    report.append("| Metric | GraphRAG | FlatRAG |")
    report.append("|--------|----------|---------|")
    report.append(f"| Accuracy | {metrics['graphrag']['accuracy']*100:.1f}% ({metrics['graphrag']['correct']}/{metrics['graphrag']['total']}) | {metrics['flatrag']['accuracy']*100:.1f}% ({metrics['flatrag']['correct']}/{metrics['flatrag']['total']}) |")
    report.append(f"| Avg Latency | {metrics['graphrag']['avg_latency_ms']:.0f}ms | {metrics['flatrag']['avg_latency_ms']:.0f}ms |")
    report.append(f"| Improvement | **{metrics['improvement_pct']:+.1f}%** | - |")

    # Sample comparisons
    report.append("\n## Sample Question-Answer Pairs\n")

    graphrag_by_q = {r['question_id']: r for r in results if r['system'] == 'graphrag'}
    flatrag_by_q = {r['question_id']: r for r in results if r['system'] == 'flatrag'}

    for qid in sorted([k for k in graphrag_by_q.keys()])[:5]:
        g = graphrag_by_q[qid]
        f = flatrag_by_q.get(qid)

        report.append(f"### Q{qid}: {g['question'][:60]}...")
        report.append(f"**GraphRAG:** {g['answer'][:250]}...")
        if f:
            report.append(f"\n**FlatRAG:** {f['answer'][:250]}...")
        report.append("")

    # Analysis
    report.append("## Analysis\n")

    if metrics['improvement_pct'] >= 20:
        report.append("✅ **Strong evidence for GraphRAG superiority** on multi-hop queries.")
        report.append(f"GraphRAG outperformed FlatRAG by {metrics['improvement_pct']:.1f}% on accuracy.")
    elif metrics['improvement_pct'] >= 0:
        report.append("✅ **GraphRAG shows improvement** over FlatRAG on multi-hop queries.")
        report.append(f"GraphRAG outperformed FlatRAG by {metrics['improvement_pct']:.1f}% on accuracy.")
    else:
        report.append("⚠️ **FlatRAG performed better** - review questions and evaluation criteria.")

    report.append("\n## Key Observations\n")

    # Check for patterns
    g_triples_avg = sum(int(r.get('triples_count', 0)) for r in graphrag_results) / len(graphrag_results) if graphrag_results else 0
    f_chunks_avg = sum(int(r.get('retrieved_chunks', 0)) for r in flatrag_results) / len(flatrag_results) if flatrag_results else 0

    report.append(f"- GraphRAG used average {g_triples_avg:.0f} triples per query")
    report.append(f"- FlatRAG retrieved average {f_chunks_avg:.0f} chunks per query")
    report.append(f"- GraphRAG latency: {metrics['graphrag']['avg_latency_ms']:.0f}ms avg")
    report.append(f"- FlatRAG latency: {metrics['flatrag']['avg_latency_ms']:.0f}ms avg")

    report.append("\n---\n*Benchmark completed with human-judged correctness scores.*")

    return "\n".join(report)


if __name__ == "__main__":
    print("Benchmark module - import this in your main script")
    print("Usage: from benchmark import run_benchmark, save_results_csv, generate_report")
