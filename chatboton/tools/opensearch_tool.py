import json
import os
from opensearchpy import OpenSearch
from .base import Connector

class OpenSearchConnector(Connector):
    def __init__(self, host=None, port=None, index=None):
        self.host = host or os.getenv("OPENSEARCH_HOST", "localhost")
        self.port = int(port or os.getenv("OPENSEARCH_PORT", "59200"))
        self.index = index or os.getenv("OPENSEARCH_INDEX", "products")
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
                index=self.index
            )
            return json.dumps(response['hits']['hits'], default=str)
        except Exception as exc:
            return f"OpenSearch error: {exc}"

class OpenSearchTool:
    def __init__(self, connector: OpenSearchConnector = None):
        self.connector = connector or OpenSearchConnector()

    def as_tool(self):
        from langchain_core.tools import tool

        @tool
        def full_text_search(query: str) -> str:
            """Performs full-text search on product names and categories in OpenSearch.
            
            Args:
                query: The search term.
            """
            body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["name", "category"]
                    }
                }
            }
            return self.connector.query(body)

        return full_text_search
