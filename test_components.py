#!/usr/bin/env python3
"""
Individual component testing script.
Run this to verify your component works before full benchmark.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_graphrag():
    """Test GraphRAG component (Dev A)."""
    print("\n" + "="*50)
    print("TESTING GRAPHRAG (Dev A)")
    print("="*50 + "\n")

    try:
        from shared import load_corpus
        from graphrag import KnowledgeGraph, GraphRAG

        # 1. Load a few documents
        print("1. Loading corpus...")
        docs = load_corpus()[:3]
        print(f"   ✓ Using {len(docs)} documents for test")

        # 2. Build graph
        print("\n2. Building knowledge graph...")
        kg = KnowledgeGraph()
        kg.build_from_documents(docs)
        stats = kg.get_stats()
        print(f"   ✓ Graph: {stats['nodes']} nodes, {stats['edges']} edges")

        # 3. Compute embeddings
        print("\n3. Computing node embeddings...")
        kg.compute_node_embeddings()
        print("   ✓ Done")

        # 4. Export graph
        print("\n4. Exporting graph...")
        Path("data/graph").mkdir(parents=True, exist_ok=True)
        kg.export_to_gexf("data/graph/test_graph.gexf")
        print("   ✓ Exported to data/graph/test_graph.gexf")

        # 5. Create GraphRAG system and test query
        print("\n5. Testing query...")
        graphrag = GraphRAG(graph=kg, traversal_depth=2, max_edges=30)

        test_questions = [
            "What relationship does Microsoft have with OpenAI?",
            "Who founded OpenAI?"
        ]

        for q in test_questions:
            print(f"\n   Q: {q}")
            answer, metadata = graphrag.query(q)
            print(f"   A: {answer[:150]}...")
            print(f"   (Triples: {metadata['triples_count']}, Latency: {metadata['latency_ms']}ms)")

        print("\n✅ GraphRAG component test PASSED")
        return True

    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flatrag():
    """Test FlatRAG component (Dev B)."""
    print("\n" + "="*50)
    print("TESTING FLATRAG (Dev B)")
    print("="*50 + "\n")

    try:
        from shared import load_corpus
        from flatrag import FlatRAG

        # 1. Load documents
        print("1. Loading corpus...")
        docs = load_corpus()[:3]
        print(f"   ✓ Using {len(docs)} documents for test")

        # 2. Build index
        print("\n2. Building ChromaDB index...")
        flatrag = FlatRAG()
        flatrag.build_index(docs, chunk_size=800)

        stats = flatrag.get_stats()
        print(f"   ✓ Index: {stats['total_chunks']} chunks")

        # 3. Test query
        print("\n3. Testing query...")
        test_questions = [
            "What relationship does Microsoft have with OpenAI?",
            "Who founded OpenAI?"
        ]

        for q in test_questions:
            print(f"\n   Q: {q}")
            answer, metadata = flatrag.query(q, k=5)
            print(f"   A: {answer[:150]}...")
            print(f"   (Retrieved: {metadata['retrieved_chunks']} chunks, Latency: {metadata['latency_ms']}ms)")

        print("\n✅ FlatRAG component test PASSED")
        return True

    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shared():
    """Test shared components."""
    print("\n" + "="*50)
    print("TESTING SHARED COMPONENTS")
    print("="*50 + "\n")

    try:
        from shared import load_corpus, load_benchmark_questions, extract_entities_from_question
        from shared.embedder import UniversalEmbedder

        # 1. Test corpus loading
        print("1. Testing corpus loader...")
        docs = load_corpus()
        print(f"   ✓ Loaded {len(docs)} documents")

        # 2. Test benchmark questions
        print("\n2. Testing benchmark questions...")
        questions = load_benchmark_questions()
        print(f"   ✓ Loaded {len(questions)} questions")
        print(f"   Sample: {questions[0]['question']}")

        # 3. Test entity extraction
        print("\n3. Testing entity extraction...")
        test_q = "Which AI companies were co-founded by former Google employees?"
        entities = extract_entities_from_question(test_q)
        print(f"   ✓ Extracted: {entities}")

        # 4. Test embedder
        print("\n4. Testing universal embedder...")
        embedder = UniversalEmbedder()
        print(f"   ✓ Provider: {embedder.config.provider}")
        print(f"   ✓ Model: {embedder.config.model_name}")

        test_texts = ["OpenAI", "Microsoft", "Anthropic"]
        embeddings = embedder.embed_documents(test_texts)
        print(f"   ✓ Generated {len(embeddings)} embeddings, dim={len(embeddings[0])}")

        print("\n✅ Shared components test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test individual components")
    parser.add_argument("--dev", choices=["A", "B"], help="Test Dev A (GraphRAG) or Dev B (FlatRAG)")
    parser.add_argument("--shared", action="store_true", help="Test shared components")
    parser.add_argument("--all", action="store_true", help="Test everything")

    args = parser.parse_args()

    if args.all:
        results = [test_shared(), test_graphrag(), test_flatrag()]
        print("\n" + "="*50)
        if all(results):
            print("✅ ALL TESTS PASSED")
            sys.exit(0)
        else:
            print("❌ SOME TESTS FAILED")
            sys.exit(1)

    elif args.dev == "A":
        sys.exit(0 if test_graphrag() else 1)

    elif args.dev == "B":
        sys.exit(0 if test_flatrag() else 1)

    elif args.shared:
        sys.exit(0 if test_shared() else 1)

    else:
        print("Usage:")
        print("  python test_components.py --shared       # Test shared utilities")
        print("  python test_components.py --dev A       # Test GraphRAG (Dev A)")
        print("  python test_components.py --dev B       # Test FlatRAG (Dev B)")
        print("  python test_components.py --all         # Test everything")
        sys.exit(1)
