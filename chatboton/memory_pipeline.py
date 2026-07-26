import asyncio
import logging
import os
import requests
from chatboton.agent import create_memory_transformer_agent, parse_json_from_llm
from chatboton.connectors.qdrant import QdrantConnector
from chatboton.connectors.opensearch import OpenSearchConnector

logger = logging.getLogger(__name__)

async def process_one_memory():
    """Fetches one short-term memory, transforms it, and moves it to long-term storage."""
    st_qdrant = QdrantConnector(collection="short_term")
    lt_qdrant = QdrantConnector(collection="long_term")
    os_connector = OpenSearchConnector(collection="long_term_memories")

    try:
        if not st_qdrant.client.collection_exists("short_term"):
            return

        # Fetch one point from short_term
        response = st_qdrant.client.scroll(
            collection_name="short_term",
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        points, _ = response
        if not points:
            return

        point = points[0]
        memory_id = point.id
        user_request = point.payload.get("request")

        if not user_request:
            # Delete invalid entry
            st_qdrant.client.delete(collection_name="short_term", points_selector=[memory_id])
            return

        logger.info(f"Processing memory {memory_id}: {user_request}")

        # Transform memory
        transformer = create_memory_transformer_agent()
        result = transformer.invoke({"messages": [{"role": "user", "content": user_request}]})
        transformed_raw = result["messages"][-1].content
        
        try:
            transformed_json = parse_json_from_llm(transformed_raw)
            include = transformed_json.get("include", False)
            transformed_content = transformed_json.get("memory", "")
        except Exception as e:
            logger.warning(f"Failed to parse memory transformer output: {e}. Raw: {transformed_raw}")
            include = False
            transformed_content = ""

        if include and transformed_content:
            # Embed transformed content
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            resp = requests.post(
                f"{ollama_url}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": transformed_content}
            )
            vector = resp.json()["embedding"]

            # Store in long-term Qdrant
            lt_qdrant.insert(vector=vector, payload={"memory": transformed_content, "original_request": user_request})

            # Store in OpenSearch
            os_connector.insert(body={"memory": transformed_content, "original_request": user_request})
            
            logger.info(f"Memory {memory_id} transformed and moved to long-term storage.")
        else:
            logger.info(f"Memory {memory_id} ignored.")

        # Delete from short-term
        st_qdrant.client.delete(collection_name="short_term", points_selector=[memory_id])

    except Exception as e:
        logger.error(f"Error in memory pipeline: {e}")

async def memory_pipeline_loop():
    """Background loop for memory processing."""
    logger.info("Starting memory pipeline background loop.")
    while True:
        try:
            await process_one_memory()
        except Exception as e:
            logger.error(f"Unexpected error in memory pipeline loop: {e}")
        await asyncio.sleep(60) # Run every minute
