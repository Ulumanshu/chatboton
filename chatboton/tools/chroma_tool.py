"""Sample vector tool: semantic search against the demo Chroma collection."""

import json
import os

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from .base import Connector


class ChromaConnector(Connector):
    def __init__(self, host=None, port=None, collection=None,
                 embed_model=None, ollama_url=None):
        self.host = host or os.getenv("CHROMA_HOST", "localhost")
        self.port = int(port or os.getenv("CHROMA_PORT", "58001"))
        self.collection = collection or os.getenv("CHROMA_COLLECTION", "product_reviews")
        self.embed_model = embed_model or os.getenv("CHROMA_EMBED_MODEL", "nomic-embed-text")
        self.ollama_url = ollama_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def _get_collection(self):
        client = chromadb.HttpClient(host=self.host, port=self.port)
        embedder = OllamaEmbeddingFunction(
            url=self.ollama_url, model_name=self.embed_model
        )
        return client.get_or_create_collection(
            name=self.collection, embedding_function=embedder
        )

    def query(self, query: str, n_results: int = 3) -> str:
        try:
            result = self._get_collection().query(
                query_texts=[query], n_results=n_results
            )
        except Exception as exc:
            return f"Chroma error: {exc}"

        hits = [
            {"document": doc, "metadata": meta, "distance": round(dist, 4)}
            for doc, meta, dist in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]
        return json.dumps(hits, default=str)


class ChromaTool:
    """Semantic search over the seeded ``product_reviews`` collection.

    Embeddings are produced by the local Ollama server (``nomic-embed-text``
    by default), so the whole stack stays offline.

    Args:
        host: Chroma server host; falls back to ``CHROMA_HOST``.
        port: Chroma server port; falls back to ``CHROMA_PORT``.
        collection: Collection name; falls back to ``CHROMA_COLLECTION``.
        embed_model: Ollama embedding model; falls back to
            ``CHROMA_EMBED_MODEL``.
        ollama_url: Ollama server URL; falls back to ``OLLAMA_BASE_URL``.
    """

    def __init__(self, connector: ChromaConnector = None):
        self.connector = connector or ChromaConnector()

    def run(self, query: str, n_results: int = 3) -> str:
        return self.connector.query(query, n_results)

    def as_tool(self):
        from langchain_core.tools import tool

        @tool
        def search_reviews(query: str, n_results: int = 3) -> str:
            """Semantically searches customer reviews of gadget-store products.

            Reviews live in a Chroma vector store; each hit carries metadata
            {product, rating}. Product names match the Postgres products
            table.

            Args:
                query: Natural-language search text.
                n_results: How many closest reviews to return.

            Returns:
                JSON list of {document, metadata, distance} hits, or an error
                description.
            """
            return self.connector.query(query, n_results)

        return search_reviews
