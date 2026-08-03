"""Hybrid search tool: search in Qdrant and OpenSearch with Reranking."""

import os
import requests
import json
from langchain_core.tools import tool
from chatboton.connectors.qdrant import QdrantConnector
from chatboton.connectors.opensearch import OpenSearchConnector

def _format_memory_with_metadata(entry: dict) -> str:
    """Formats a memory with its stored metadata tags as extra context."""
    memory = entry.get("memory")
    if not memory:
        return ""
    context_bits = []
    if entry.get("object") or entry.get("subject"):
        about = " — ".join(bit for bit in (entry.get("object"), entry.get("subject")) if bit)
        context_bits.append(f"About: {about}.")
    if entry.get("sentiment"):
        context_bits.append(f"Sentiment: {entry['sentiment']}.")
    for field in ("topics", "technologies", "tags"):
        values = entry.get(field) or []
        if isinstance(values, str):
            values = [values]
        if values:
            context_bits.append(f"{field.capitalize()}: " + ", ".join(str(v) for v in values) + ".")
    if context_bits:
        return f"{memory} [{' '.join(context_bits)}]"
    return memory


@tool
def search_long_term_memory(query: str, limit: int = 3) -> str:
    """Searches long-term memory using hybrid search (Vector + BM25) and re-ranks results.
    
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
        memory = _format_memory_with_metadata(hit.get("payload", {}))
        if memory:
            memories.add(memory)

    for hit in opensearch_results:
        memory = _format_memory_with_metadata(hit.get("_source", {}))
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
