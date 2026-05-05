"""
FlatRAG Implementation - Dev B
Using ChromaDB for vector similarity search.
"""

from typing import List, Dict, Any, Tuple
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document

from shared import (
    llm_generate,
    load_corpus,
    save_results_csv,
    load_benchmark_questions
)
from shared.embedder import UniversalEmbedder


# ============================================
# Document Processing
# ============================================

class DocumentChunker:
    """Split documents into chunks for vector storage."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Split documents into chunks with metadata.
        Returns list of LangChain Document objects.
        """
        all_chunks = []

        for doc in documents:
            # Split text
            doc_chunks = self.splitter.split_text(doc['content'])

            for i, chunk_text in enumerate(doc_chunks):
                chunk = Document(
                    page_content=chunk_text,
                    metadata={
                        "doc_id": doc['doc_id'],
                        "title": doc['title'],
                        "chunk_id": f"{doc['doc_id']}_{i}",
                        "chunk_index": i,
                        "total_chunks": len(doc_chunks)
                    }
                )
                all_chunks.append(chunk)

        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


# ============================================
# ChromaDB Vector Store
# ============================================

class FlatRAG:
    """Flat RAG system using ChromaDB vector store."""

    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "flatrag",
        embedding_function = None
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = embedding_function or UniversalEmbedder()
        self.vectorstore = None
        self.chunks = []

    def build_index(self, documents: List[Dict[str, Any]], chunk_size: int = 1000):
        """
        Build ChromaDB vector index from documents.
        """
        print("\n" + "="*60)
        print("BUILDING FLATRAG INDEX")
        print("="*60)

        # 1. Chunk documents
        print(f"\n1. Chunking documents (size={chunk_size}, overlap=200)...")
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=200)
        self.chunks = chunker.chunk_documents(documents)

        # 2. Create embeddings and store in Chroma
        print(f"\n2. Creating embeddings and storing in ChromaDB...")
        print(f"   Embedder: {self.embedder.config.provider} / {self.embedder.config.model_name}")

        # Use Chroma with custom embedding function
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder,
            persist_directory=self.persist_dir
        )

        # Add documents in batches
        batch_size = 100
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i+batch_size]
            texts = [chunk.page_content for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]

            self.vectorstore.add_texts(texts, metadatas)
            print(f"   Added batch {i//batch_size + 1}/{(len(self.chunks)-1)//batch_size + 1}")

        print(f"\n✓ Index built: {self.vectorstore._collection.count()} vectors stored")
        return self

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k relevant chunks with similarity scores.
        Returns list of (document, score) tuples.
        """
        if not self.vectorstore:
            raise ValueError("Index not built. Call build_index() first.")

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=filter_metadata
        )

        return results

    def query(self, question: str, k: int = 5) -> Tuple[str, Dict[str, Any]]:
        """
        Execute FlatRAG query pipeline.
        Returns (answer, metadata)
        """
        start_time = time.time()
        metadata = {
            "retrieved_chunks": 0,
            "total_context_chars": 0
        }

        # 1. Retrieve relevant chunks
        results = self.retrieve(question, k=k)
        metadata["retrieved_chunks"] = len(results)

        if not results:
            return "I couldn't find relevant information.", metadata

        # 2. Build context
        context_parts = []
        for doc, score in results:
            context_parts.append(f"[Relevance: {score:.2f}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)
        metadata["total_context_chars"] = len(context)

        # 3. Generate answer
        prompt = f"""
Answer the question using ONLY the provided context from documents.

Rules:
- Base your answer ONLY on the provided context
- If information is insufficient, say "I don't have enough information"
- Do not make up facts
- Cite which document chunks you used if possible

Question: {question}

Context:
{context}

Answer:
"""

        answer = llm_generate(prompt, max_tokens=500)
        latency = int((time.time() - start_time) * 1000)
        metadata["latency_ms"] = latency

        return answer, metadata

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the index."""
        if not self.vectorstore:
            return {"error": "Index not built"}

        return {
            "total_chunks": len(self.chunks),
            "index_size": self.vectorstore._collection.count(),
            "persist_dir": self.persist_dir,
            "embedder": f"{self.embedder.config.provider}/{self.embedder.config.model_name}"
        }

    def delete_index(self):
        """Delete the persisted index."""
        import shutil
        if self.vectorstore:
            self.vectorstore = None
        if Path(self.persist_dir).exists():
            shutil.rmtree(self.persist_dir)
            print(f"Deleted index at {self.persist_dir}")


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("FLATRAG SYSTEM TEST (Dev B)")
    print("="*60)

    # Load corpus
    documents = load_corpus()
    print(f"Loaded {len(documents)} documents")

    if len(documents) == 0:
        print("ERROR: No documents loaded. Check corpus/ directory.")
        exit(1)

    # Build index
    flatrag = FlatRAG()
    flatrag.build_index(documents[:min(10, len(documents))])  # Test with first 10

    # Show stats
    stats = flatrag.get_stats()
    print(f"\nIndex stats: {stats}")

    # Test queries
    test_questions = [
        "What relationship does Microsoft have with OpenAI?",
        "Who founded OpenAI?",
        "What cloud providers do AI companies use?"
    ]

    print("\n" + "="*60)
    print("TEST QUERIES")
    print("="*60)

    for question in test_questions:
        print(f"\nQ: {question}")
        answer, metadata = flatrag.query(question, k=5)
        print(f"A: {answer[:200]}...")
        print(f"  Retrieved: {metadata['retrieved_chunks']} chunks")
        print(f"  Context: {metadata['total_context_chars']} chars")
        print(f"  Latency: {metadata['latency_ms']}ms")

    print("\n" + "="*60)
    print("FlatRAG test complete!")
    print("="*60)
