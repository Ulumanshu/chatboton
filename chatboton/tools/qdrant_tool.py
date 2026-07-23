import json
import os
from typing import Optional, List

from qdrant_client import QdrantClient
from qdrant_client.http import models
from .base import Connector

class QdrantConnector(Connector):
    def __init__(self, host=None, port=None, collection=None):
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = int(port or os.getenv("QDRANT_PORT", "58333"))
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "products")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    def query(self, vector: List[float], limit: int = 3) -> str:
        try:
            results = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=limit
            ).points
            hits = [
                {"id": hit.id, "payload": hit.payload, "score": hit.score}
                for hit in results
            ]
            return json.dumps(hits, default=str)
        except Exception as exc:
            return f"Qdrant error: {exc}"

class QdrantTool:
    def __init__(self, connector: QdrantConnector = None):
        self.connector = connector or QdrantConnector()

    def as_tool(self):
        from langchain_core.tools import tool
        import requests

        @tool
        def search_products_v2(query: str, limit: int = 3) -> str:
            """Searches products using vector similarity in Qdrant.
            
            Args:
                query: Natural language query.
                limit: Number of results.
            """
            # Simple way to get embeddings - use Ollama if available or a mock for now
            # In a real scenario, we'd use the same embedding function as seed.py
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            try:
                resp = requests.post(
                    f"{ollama_url}/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": query}
                )
                vector = resp.json()["embedding"]
            except Exception as e:
                return f"Embedding error: {e}"

            return self.connector.query(vector, limit)

        return search_products_v2
