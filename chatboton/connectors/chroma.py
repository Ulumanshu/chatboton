import json
import os
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from .base import Connector

class ChromaConnector(Connector):
    def __init__(self, collection, host=None, port=None,
                 embed_model=None, ollama_url=None):
        self.host = host or os.getenv("CHROMA_HOST", "localhost")
        self.port = int(port or os.getenv("CHROMA_PORT", "58001"))
        self.collection = collection
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

    def insert(self, document: str, metadata: dict = None) -> str:
        try:
            import uuid
            self._get_collection().add(
                documents=[document],
                metadatas=[metadata] if metadata else [{}],
                ids=[str(uuid.uuid4())]
            )
            return "Success"
        except Exception as exc:
            return f"Chroma error: {exc}"
