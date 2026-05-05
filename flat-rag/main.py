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

    # Nếu reindex được chỉ định, hoặc DB chưa tồn tại, thì build lại
    if args.reindex or not os.path.exists(args.persist):
        print("Indexing files and building vectorstore...")
        retriever = build_or_load_vectorstore(data_dir=args.data, persist_directory=args.persist)
    else:
        # Thực tế Chroma.from_documents sẽ load DB nếu tồn tại, nhưng để đơn giản, rebuild is recommended
        print("Opening existing vectorstore by re-building from data (recommended). Use --reindex to force fresh index")
        retriever = build_or_load_vectorstore(data_dir=args.data, persist_directory=args.persist)

    chain = create_retrieval_chain(retriever)

    if args.query:
        ans = chain.run(args.query)
        print("\nAnswer:\n", ans)
        return

    print("Enter queries (empty line to quit):")
    while True:
        try:
            q = input("Q> ")
        except EOFError:
            break
        if not q.strip():
            break
        a = chain.run(q)
        print("A> ", a)


if __name__ == "__main__":
    main()
