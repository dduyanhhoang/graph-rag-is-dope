import os
import argparse
from dotenv import load_dotenv

from utils import build_or_load_vectorstore, create_retrieval_chain


# Load environment variables from .env
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Flat RAG demo: ingest local PDF/TXT and answer queries.")
    parser.add_argument("--data", type=str, default="data", help="Data directory containing .pdf/.txt")
    parser.add_argument("--persist", type=str, default=".chromadb", help="Chroma persist directory")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing of files")
    parser.add_argument("--query", type=str, help="Optional single query to run")
    args = parser.parse_args()

    if args.reindex:
        print("Re-indexing files and rebuilding vectorstore...")
    else:
        print("Loading existing vectorstore if available; otherwise building it from data...")

    retriever = build_or_load_vectorstore(
        data_dir=args.data,
        persist_directory=args.persist,
        rebuild=args.reindex,
    )

    chain = create_retrieval_chain(retriever)

    if args.query:
        result = chain.invoke({"input": args.query})
        print("\nAnswer:\n", result["answer"])
        sources = sorted({item.metadata.get("source", "unknown") for item in result.get("context", [])})
        if sources:
            print("\nSources:\n" + "\n".join(f"- {source}" for source in sources))
        return

    print("Enter queries (empty line to quit):")
    while True:
        try:
            q = input("Q> ")
        except EOFError:
            break
        if not q.strip():
            break
        result = chain.invoke({"input": q})
        print("A> ", result["answer"])
        sources = sorted({item.metadata.get("source", "unknown") for item in result.get("context", [])})
        if sources:
            print("Sources:")
            for source in sources:
                print(f"- {source}")


if __name__ == "__main__":
    main()
