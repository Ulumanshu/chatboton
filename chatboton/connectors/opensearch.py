import json
import os
from opensearchpy import OpenSearch
from .base import Connector

class OpenSearchConnector(Connector):
    def __init__(self, collection, host=None, port=None):
        self.host = host or os.getenv("OPENSEARCH_HOST", "localhost")
        self.port = int(port or os.getenv("OPENSEARCH_PORT", "59200"))
        self.collection = collection
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = OpenSearch(
                hosts=[{'host': self.host, 'port': self.port}],
                use_ssl=False,
                verify_certs=False,
            )
        return self._client

    def query(self, body: dict) -> str:
        try:
            response = self.client.search(
                body=body,
                index=self.collection
            )
            return json.dumps(response['hits']['hits'], default=str)
        except Exception as exc:
            return f"OpenSearch error: {exc}"

    def insert(self, body: dict) -> str:
        try:
            response = self.client.index(
                index=self.collection,
                body=body,
                refresh=True
            )
            return json.dumps(response, default=str)
        except Exception as exc:
            return f"OpenSearch error: {exc}"
