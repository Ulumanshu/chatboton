import pytest
import asyncio
from unittest.mock import MagicMock, patch
from chatboton.memory_pipeline import process_one_memory
import json
from langchain_core.messages import AIMessage

@pytest.mark.asyncio
async def test_process_one_memory_ignore_mechanism(monkeypatch):
    # Mock QdrantConnectors
    mock_st_qdrant = MagicMock()
    mock_lt_qdrant = MagicMock()
    mock_os_connector = MagicMock()

    def mock_qdrant_factory(collection):
        if collection == "short_term":
            return mock_st_qdrant
        return mock_lt_qdrant

    monkeypatch.setattr("chatboton.memory_pipeline.QdrantConnector", mock_qdrant_factory)
    monkeypatch.setattr("chatboton.memory_pipeline.OpenSearchConnector", lambda collection: mock_os_connector)

    # Mock short-term memory content
    mock_point = MagicMock()
    mock_point.id = "123"
    mock_point.payload = {"request": "hey whats my name?"}
    mock_st_qdrant.client.scroll.return_value = ([mock_point], None)
    mock_st_qdrant.client.collection_exists.return_value = True

    # Mock Memory Transformer Agent
    mock_transformer = MagicMock()
    # Case 1: IGNORE via include=False
    mock_transformer.invoke.return_value = {
        "messages": [AIMessage(content='{"include": false, "memory": "ignored"}')]
    }
    
    with patch("chatboton.memory_pipeline.create_memory_transformer_agent", return_value=mock_transformer):
        await process_one_memory()

    # Verify lt_qdrant.insert was NOT called
    assert mock_lt_qdrant.insert.call_count == 0
    # Verify it was deleted from short-term
    mock_st_qdrant.client.delete.assert_called_with(collection_name="short_term", points_selector=["123"])

@pytest.mark.asyncio
async def test_process_one_memory_include_mechanism(monkeypatch):
    # Mock QdrantConnectors
    mock_st_qdrant = MagicMock()
    mock_lt_qdrant = MagicMock()
    mock_os_connector = MagicMock()

    def mock_qdrant_factory(collection):
        if collection == "short_term":
            return mock_st_qdrant
        return mock_lt_qdrant

    monkeypatch.setattr("chatboton.memory_pipeline.QdrantConnector", mock_qdrant_factory)
    monkeypatch.setattr("chatboton.memory_pipeline.OpenSearchConnector", lambda collection: mock_os_connector)

    # Mock short-term memory content
    mock_point = MagicMock()
    mock_point.id = "456"
    mock_point.payload = {"request": "I am working on a React project."}
    mock_st_qdrant.client.scroll.return_value = ([mock_point], None)
    mock_st_qdrant.client.collection_exists.return_value = True

    # Mock Memory Transformer Agent
    mock_transformer = MagicMock()
    # Case 2: INCLUDE via include=True
    mock_transformer.invoke.return_value = {
        "messages": [AIMessage(content='{"include": true, "memory": "User is working on a React project."}')]
    }
    
    # Mock embeddings response
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embedding": [0.1] * 768}
    monkeypatch.setattr("requests.post", lambda url, json: mock_resp)

    with patch("chatboton.memory_pipeline.create_memory_transformer_agent", return_value=mock_transformer):
        await process_one_memory()

    # Verify lt_qdrant.insert WAS called
    assert mock_lt_qdrant.insert.call_count == 1
    args, kwargs = mock_lt_qdrant.insert.call_args
    assert kwargs["payload"]["memory"] == "User is working on a React project."
    
    # Verify it was deleted from short-term
    mock_st_qdrant.client.delete.assert_called_with(collection_name="short_term", points_selector=["456"])
