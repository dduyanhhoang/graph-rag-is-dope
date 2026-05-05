#!/usr/bin/env python3
"""
Embedding model configuration and factory.
Supports both OpenAI and Sentence Transformers for universal use across GraphRAG and Flat RAG.
"""

import os
from typing import List
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    provider: str  # "openai" or "sentence_transformers"
    model_name: str
    dimension: int
    use_openai: bool


def get_embedding_config() -> EmbeddingConfig:
    """
    Get embedding configuration from environment variables.
    Returns a config object that works for both GraphRAG and Flat RAG.
    """
    use_openai = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    if use_openai:
        # OpenAI embeddings
        dimension = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }.get(model_name, 1536)
        return EmbeddingConfig(
            provider="openai",
            model_name=model_name,
            dimension=dimension,
            use_openai=True
        )
    else:
        # Sentence Transformers (local, free)
        dimension = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-MiniLM-L6-v2": 384,
            "multi-qa-MiniLM-L6-dot-v1": 384,
        }.get(model_name, 384)
        return EmbeddingConfig(
            provider="sentence_transformers",
            model_name=model_name,
            dimension=dimension,
            use_openai=False
        )


class UniversalEmbedder:
    """
    Universal embedding interface that works with both OpenAI and Sentence Transformers.
    Use this for BOTH GraphRAG (node embeddings) and Flat RAG (chunk embeddings).
    """

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or get_embedding_config()
        self._model = None
        self._client = None

    def _get_client(self):
        """Lazy initialization of embedding client."""
        if self._model is not None or self._client is not None:
            return self._client if self.config.use_openai else self._model

        if self.config.use_openai:
            # OpenAI embeddings
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            # Sentence Transformers (local)
            from sentence_transformers import SentenceTransformer
            print(f"Loading sentence transformer model: {self.config.model_name}")
            self._model = SentenceTransformer(self.config.model_name)

        return self._client if self.config.use_openai else self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Works for both document chunks (Flat RAG) and node labels/descriptions (GraphRAG).
        """
        client = self._get_client()

        if self.config.use_openai:
            # OpenAI: batch up to 2048 texts per request
            embeddings = []
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = client.embeddings.create(
                    model=self.config.model_name,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            return embeddings
        else:
            # Sentence Transformers
            embeddings = client.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10
            )
            return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        client = self._get_client()

        if self.config.use_openai:
            response = client.embeddings.create(
                model=self.config.model_name,
                input=[text]
            )
            return response.data[0].embedding
        else:
            embedding = client.encode([text], convert_to_numpy=True)
            return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.config.dimension


def test_embedder():
    """Quick test of the universal embedder."""
    print("Testing Universal Embedder...")
    embedder = UniversalEmbedder()

    test_texts = [
        "OpenAI was founded by Sam Altman.",
        "Microsoft invested in OpenAI.",
        "Anthropic develops Claude."
    ]

    print(f"Provider: {embedder.config.provider}")
    print(f"Model: {embedder.config.model_name}")
    print(f"Dimension: {embedder.dimension}")

    embeddings = embedder.embed_documents(test_texts)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"First embedding dimension: {len(embeddings[0])}")

    # Test query embedding
    query_embedding = embedder.embed_query("Who founded OpenAI?")
    print(f"Query embedding dimension: {len(query_embedding)}")

    print("Embedder test complete!")


if __name__ == "__main__":
    test_embedder()
