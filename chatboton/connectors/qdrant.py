import json
import os
from typing import List
from qdrant_client import QdrantClient
from .base import Connector

class QdrantConnector(Connector):
    def __init__(self, collection, host=None, port=None):
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = int(port or os.getenv("QDRANT_PORT", "58333"))
        self.collection = collection
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

    def insert(self, vector: List[float], payload: dict) -> str:
        try:
            import uuid
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    {
                        "id": str(uuid.uuid4()),
                        "vector": vector,
                        "payload": payload
                    }
                ]
            )
            return "Success"
        except Exception as exc:
            return f"Qdrant error: {exc}"
