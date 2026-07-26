"""Hybrid search tool: search in Qdrant and OpenSearch with Reranking."""

import os
import requests
import json
from langchain_core.tools import tool
from chatboton.connectors.qdrant import QdrantConnector
from chatboton.connectors.opensearch import OpenSearchConnector

@tool
def search_long_term_memory(query: str, limit: int = 3) -> str:
    """Searches long-term memory using hybrid search (Vector + BM25) and re-ranks results.
    This tool MUST be called FIRST for EVERY user interaction when its seems that there is not enough information
    to retrieve context.
    
    Args:
        query: Natural language query.
        limit: Number of results.
    """
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # 1. Vector Search (Qdrant)
    try:
        resp = requests.post(
            f"{ollama_url}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": query}
        )
        vector = resp.json()["embedding"]
    except Exception as e:
        return f"Embedding error: {e}"

    qdrant_results_raw = QdrantConnector(collection="long_term").query(vector, limit)
    try:
        qdrant_results = json.loads(qdrant_results_raw)
    except Exception:
        qdrant_results = []

    # 2. BM25 Search (OpenSearch)
    os_connector = OpenSearchConnector(collection="long_term_memories")
    opensearch_results_raw = os_connector.text_search(query, limit)
    try:
        opensearch_results = json.loads(opensearch_results_raw)
    except Exception:
        opensearch_results = []

    # 3. Combine results
    memories = set()
    for hit in qdrant_results:
        payload = hit.get("payload", {})
        memory = payload.get("memory")
        if memory:
            memories.add(memory)
    
    for hit in opensearch_results:
        source = hit.get("_source", {})
        memory = source.get("memory")
        if memory:
            memories.add(memory)

    if not memories:
        return json.dumps([])

    # 4. Rerank results
    from chatboton.agent import create_reranker_agent
    reranker = create_reranker_agent()
    rerank_input = f"Query: {query}\nMemories:\n" + "\n".join([f"- {m}" for m in memories])
    
    try:
        rerank_result = reranker.invoke({"messages": [{"role": "user", "content": rerank_input}]})
        reranked_memories_raw = rerank_result["messages"][-1].content
        
        # Try to parse JSON from the response
        try:
            # The agent is instructed to return ONLY JSON, but let's be safe
            start = reranked_memories_raw.find("[")
            end = reranked_memories_raw.rfind("]") + 1
            if start != -1 and end != 0:
                reranked_memories = json.loads(reranked_memories_raw[start:end])
                return json.dumps(reranked_memories[:limit])
            else:
                # Fallback if JSON parsing fails
                return json.dumps(list(memories)[:limit])
        except Exception:
            return json.dumps(list(memories)[:limit])
            
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Reranker error: {e}")
        return json.dumps(list(memories)[:limit])
