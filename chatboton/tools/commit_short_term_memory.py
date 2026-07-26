import os
import requests
from langchain_core.tools import tool
from chatboton.connectors.qdrant import QdrantConnector


@tool
def commit_short_term_memory(user_request: str) -> str:
    """Stores ALL user requests as searchable vectors in Qdrant.

    Args:
        user_request: Natural language request.

    """
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        resp = requests.post(
            f"{ollama_url}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": user_request}
        )
        vector = resp.json()["embedding"]
    except Exception as e:
        return f"Embedding error: {e}"

    return QdrantConnector(collection="short_term").insert(vector=vector, payload={"request": user_request})
