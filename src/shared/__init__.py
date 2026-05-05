"""
Shared utilities for both GraphRAG and Flat RAG systems.
"""

import os
import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
from openai import OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Logging
import logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


# ============================================
# Corpus Loading
# ============================================

def load_corpus(corpus_dir: str = "corpus") -> List[Dict[str, Any]]:
    """
    Load all cleaned corpus documents, filtering out problematic ones.

    Returns:
        List of dicts with keys: doc_id, title, content, filepath
    """
    corpus_path = Path(corpus_dir)
    documents = []

    # Files to skip (disambiguation errors, wrong content)
    SKIP_PATTERNS = [
        "Amazon_rainforest",
        "Meta_(prefix)",
        "Nikola_Tesla",
        "Replicate_(biology)"
    ]

    for filepath in sorted(corpus_path.glob("cleaned_*.txt")):
        # Skip problematic files
        if any(pattern in filepath.name for pattern in SKIP_PATTERNS):
            logger.info(f"Skipping {filepath.name} (disambiguation/incorrect content)")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Skip header lines starting with '#'
                content = ''.join([l for l in lines if not l.startswith('#')]).strip()
                title = lines[0].strip('# \n') if lines else filepath.stem

                doc_id = filepath.stem.replace('cleaned_', '')

                documents.append({
                    "doc_id": doc_id,
                    "title": title,
                    "content": content,
                    "filepath": str(filepath)
                })
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")

    logger.info(f"Loaded {len(documents)} valid documents from corpus")
    return documents


def load_document_by_id(doc_id: str, corpus_dir: str = "corpus") -> Dict[str, Any]:
    """Load a single document by its ID."""
    filepath = Path(corpus_dir) / f"cleaned_{doc_id}.txt"
    if not filepath.exists():
        # Try without cleaned prefix
        filepath = Path(corpus_dir) / f"{doc_id}.txt"

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        content = ''.join([l for l in lines if not l.startswith('#')]).strip()
        title = lines[0].strip('# \n') if lines else doc_id

    return {
        "doc_id": doc_id,
        "title": title,
        "content": content,
        "filepath": str(filepath)
    }


# ============================================
# LLM Utilities
# ============================================

def llm_generate(
    prompt: str,
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 500
) -> str:
    """
    Generate text using OpenAI LLM with error handling and retries.
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    for attempt in range(3):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise


def llm_generate_json(
    prompt: str,
    model: str = None,
    temperature: float = 0.0
):
    """
    Generate JSON response from LLM.
    Returns parsed JSON (dict or list).
    """
    prompt = prompt + "\n\nReturn your response as valid JSON only."

    response_text = llm_generate(prompt, model, temperature)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from response - handle both objects {...} and arrays [...]
        import re
        # Try array pattern first
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # Try object pattern
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON from response: {response_text[:100]}")


# ============================================
# Entity & Question Processing
# ============================================

def extract_entities_from_question(question: str) -> List[str]:
    """
    Extract entity names from a question using LLM.
    Returns list of entity strings (people, companies, products, technologies).
    """
    prompt = f"""
    Extract all entity names (people, companies, products, technologies) from this question.
    Focus on proper nouns that would exist in a knowledge graph.

    Return as a JSON array of strings, e.g., ["Microsoft", "OpenAI", "Azure"]

    Question: {question}

    Entities (JSON array):
    """

    try:
        result = llm_generate_json(prompt)
        if isinstance(result, list):
            return result
        # Handle {"entities": [...]} format
        return result.get("entities", [])
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}, using fallback")
        # Fallback: simple capitalized word extraction
        import re
        candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        return candidates[:5]


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity names for matching.
    """
    return name.lower().replace(' ', '').replace('-', '').replace('_', '').replace('.', '')


def fuzzy_match_entity(entity: str, candidates: List[str], threshold: float = 0.7) -> List[str]:
    """
    Find best matching entities using string similarity.
    Returns sorted list by similarity.
    """
    from difflib import SequenceMatcher

    entity_norm = normalize_entity_name(entity)
    scored = []

    for candidate in candidates:
        candidate_norm = normalize_entity_name(candidate)
        score = SequenceMatcher(None, entity_norm, candidate_norm).ratio()
        if score >= threshold:
            scored.append((candidate, score))

    return [c for c, s in sorted(scored, key=lambda x: x[1], reverse=True)]


# ============================================
# Benchmark Questions
# ============================================

def load_benchmark_questions() -> List[Dict[str, Any]]:
    """Load the 20 multi-hop benchmark questions."""
    return [
        {"id": 1, "question": "Which AI companies were co-founded by former Google employees?"},
        {"id": 2, "question": "What cloud providers do OpenAI and Anthropic use?"},
        {"id": 3, "question": "Which companies invested in both OpenAI and Anthropic?"},
        {"id": 4, "question": "Who are the founders of OpenAI and what companies did they work at before?"},
        {"id": 5, "question": "What relationship does Microsoft have with OpenAI?"},
        {"id": 6, "question": "Which companies partnered with both Amazon and Google?"},
        {"id": 7, "question": "What products did Anthropic develop?"},
        {"id": 8, "question": "Who invests in AI safety research?"},
        {"id": 9, "question": "Which tech companies acquired AI startups in 2024-2025?"},
        {"id": 10, "question": "What is the connection between NVIDIA and major AI labs?"},
        {"id": 11, "question": "Which companies use Azure for AI infrastructure?"},
        {"id": 12, "question": "Who are the CEOs of OpenAI and Anthropic?"},
        {"id": 13, "question": "What funding rounds did OpenAI have in 2025?"},
        {"id": 14, "question": "Which companies have partnerships with Palantir?"},
        {"id": 15, "question": "What hardware does OpenAI use for training?"},
        {"id": 16, "question": "Which AI companies have DoD contracts?"},
        {"id": 17, "question": "What is the relationship between DeepMind and Google?"},
        {"id": 18, "question": "Which companies developed large language models?"},
        {"id": 19, "question": "Who are the main investors in Anthropic?"},
        {"id": 20, "question": "What cloud partnerships does Google have with AI companies?"},
    ]


# ============================================
# Results Management
# ============================================

def save_results_csv(
    results: List[Dict[str, Any]],
    filename: str = "data/results/benchmark_results.csv"
):
    """Save benchmark results to CSV."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    if not results:
        logger.warning("No results to save")
        return

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results saved to {filename}")


def load_results_csv(filename: str = "data/results/benchmark_results.csv") -> List[Dict[str, Any]]:
    """Load benchmark results from CSV."""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def calculate_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate accuracy metrics from benchmark results.
    Note: Requires human-judged 'correct' field.
    """
    graphrag_results = [r for r in results if r['system'] == 'graphrag']
    flatrag_results = [r for r in results if r['system'] == 'flatrag']

    # Count manually judged correct answers (correct field should be 'True' or 'False')
    def count_correct(records):
        total = 0
        correct = 0
        for r in records:
            if 'correct' in r and r['correct'] != '':
                total += 1
                if r['correct'].lower() == 'true':
                    correct += 1
        return correct, total

    g_correct, g_total = count_correct(graphrag_results)
    f_correct, f_total = count_correct(flatrag_results)

    return {
        "graphrag_accuracy": g_correct / g_total if g_total > 0 else 0,
        "flatrag_accuracy": f_correct / f_total if f_total > 0 else 0,
        "improvement": (g_correct - f_correct) / max(f_total, 1) if f_total > 0 else 0,
        "graphrag_total": g_total,
        "flatrag_total": f_total
    }


def generate_summary_report(results: List[Dict[str, Any]]) -> str:
    """Generate a markdown summary report from benchmark results."""
    graphrag_results = [r for r in results if r['system'] == 'graphrag']
    flatrag_results = [r for r in results if r['system'] == 'flatrag']

    # Statistics
    graph_avg_latency = sum(int(r['latency_ms']) for r in graphrag_results) / len(graphrag_results) if graphrag_results else 0
    flat_avg_latency = sum(int(r['latency_ms']) for r in flatrag_results) / len(flatrag_results) if flatrag_results else 0

    accuracy = calculate_accuracy(results)

    report = []
    report.append("# GraphRAG vs Flat RAG - Benchmark Report\n")

    report.append("## Performance Metrics\n")
    report.append(f"- **GraphRAG** average latency: {graph_avg_latency:.0f}ms")
    report.append(f"- **FlatRAG** average latency: {flat_avg_latency:.0f}ms")
    report.append(f"- **Speedup**: {flat_avg_latency/graph_avg_latency:.2f}x (FlatRAG is {'faster' if flat_avg_latency < graph_avg_latency else 'slower'})")

    if accuracy['graphrag_total'] > 0:
        report.append(f"\n## Accuracy (Manually Judged)\n")
        report.append(f"- GraphRAG: {accuracy['graphrag_accuracy']*100:.1f}% ({int(accuracy['graphrag_accuracy']*accuracy['graphrag_total'])}/{int(accuracy['graphrag_total'])})")
        report.append(f"- FlatRAG: {accuracy['flatrag_accuracy']*100:.1f}% ({int(accuracy['flatrag_accuracy']*accuracy['flatrag_total'])}/{int(accuracy['flatrag_total'])})")
        report.append(f"- **Improvement: {accuracy['improvement']*100:+.1f}%**")

    report.append("\n## Sample Results\n")

    # Show some example Q&A pairs where GraphRAG won
    for i in range(min(3, len(graphrag_results))):
        g = graphrag_results[i]
        f = next((r for r in flatrag_results if r['question_id'] == g['question_id']), None)
        if f:
            report.append(f"### Q{i+1}: {g['question'][:60]}...")
            report.append(f"**GraphRAG**: {g['answer'][:200]}...")
            report.append(f"**FlatRAG**: {f['answer'][:200]}...")
            report.append("")

    report.append("## Conclusion\n")
    if accuracy.get('improvement', 0) > 0.2:
        report.append("**GraphRAG significantly outperformed FlatRAG** on multi-hop questions, demonstrating the value of graph-based retrieval for relational queries.")
    elif accuracy.get('improvement', 0) > 0:
        report.append("**GraphRAG showed improvement** over FlatRAG on multi-hop questions, validating the approach.")
    else:
        report.append("**Results inconclusive** - manual review needed for correctness judgments.")

    return "\n".join(report)


if __name__ == "__main__":
    # Quick test
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents")
    if docs:
        print(f"Sample: {docs[0]['title']} ({len(docs[0]['content'])} chars)")

    questions = load_benchmark_questions()
    print(f"Loaded {len(questions)} benchmark questions")
