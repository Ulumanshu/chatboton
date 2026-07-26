"""Vector tool: search in Qdrant long-term memory."""

import os
import requests
from langchain_core.tools import tool
from chatboton.connectors.qdrant import QdrantConnector

@tool
def search_long_term_memory(query: str, limit: int = 3) -> str:
    """Searches long-term memory using vector similarity in Qdrant.
    
    Args:
        query: Natural language query.
        limit: Number of results.
    """
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        resp = requests.post(
            f"{ollama_url}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": query}
        )
        vector = resp.json()["embedding"]
    except Exception as e:
        return f"Embedding error: {e}"

    return QdrantConnector(collection="long_term").query(vector, limit)
