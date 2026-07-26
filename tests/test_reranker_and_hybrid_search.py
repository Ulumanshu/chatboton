import json
from unittest.mock import MagicMock, patch
import pytest
from chatboton.connectors.opensearch import OpenSearchConnector
from chatboton.tools.search_long_term_memory import search_long_term_memory
from chatboton.agent import create_reranker_agent

class TestHybridSearchAndReranker:

    def test_opensearch_text_search(self):
        client = MagicMock()
        client.indices.exists.return_value = True
        client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"memory": "User likes Python", "original_request": "I love python"}}
                ]
            }
        }
        
        connector = OpenSearchConnector(collection="long_term_memories", host="h", port=1)
        connector._client = client
        
        result = connector.text_search("Python", limit=1)
        hits = json.loads(result)
        assert len(hits) == 1
        assert hits[0]["_source"]["memory"] == "User likes Python"
        
        client.search.assert_called_once()
        body = client.search.call_args[1]["body"]
        assert "query" in body
        assert "match" in body["query"]
        assert body["query"]["match"]["memory"] == "Python"

    @patch("requests.post")
    @patch("chatboton.connectors.qdrant.QdrantConnector.query")
    @patch("chatboton.connectors.opensearch.OpenSearchConnector.text_search")
    def test_search_long_term_memory_hybrid(self, mock_os_search, mock_qdrant_query, mock_post):
        # We need to mock create_reranker_agent where it is imported: inside search_long_term_memory
        # Since it is a local import, we patch 'chatboton.agent.create_reranker_agent'
        with patch("chatboton.agent.create_reranker_agent") as mock_create_reranker:
            # Mock embeddings
            mock_post.return_value.json.return_value = {"embedding": [0.1] * 768}
            
            # Mock Qdrant results
            mock_qdrant_query.return_value = json.dumps([
                {"payload": {"memory": "Vector memory 1"}},
                {"payload": {"memory": "Common memory"}}
            ])
            
            # Mock OpenSearch results
            mock_os_search.return_value = json.dumps([
                {"_source": {"memory": "BM25 memory 1"}},
                {"_source": {"memory": "Common memory"}}
            ])
            
            # Mock Reranker Agent
            mock_reranker = MagicMock()
            mock_create_reranker.return_value = mock_reranker
            mock_reranker.invoke.return_value = {
                "messages": [
                    MagicMock(content='["Common memory", "Vector memory 1", "BM25 memory 1"]')
                ]
            }
            
            result = search_long_term_memory.invoke({"query": "test query", "limit": 3})
            
            final_memories = json.loads(result)
            assert len(final_memories) == 3
            # In the mock, it doesn't matter much the order, but we can check if all are there
            assert "Common memory" in final_memories
            assert "Vector memory 1" in final_memories
            assert "BM25 memory 1" in final_memories
            
            # Verify Reranker was called with combined unique memories
            # LangChain agent invoke calls can be complex, let's just check if it was called
            mock_reranker.invoke.assert_called_once()
            reranker_input_call = mock_reranker.invoke.call_args[0][0]
            reranker_input = reranker_input_call["messages"][0]["content"]
            assert "Common memory" in reranker_input
            assert "Vector memory 1" in reranker_input
            assert "BM25 memory 1" in reranker_input
            # Check uniqueness (Common memory should appear once in the list passed to reranker)
            assert reranker_input.count("- Common memory") == 1

    @patch("requests.post")
    @patch("chatboton.connectors.qdrant.QdrantConnector.query")
    @patch("chatboton.connectors.opensearch.OpenSearchConnector.text_search")
    def test_search_long_term_memory_reranker_fallback(self, mock_os_search, mock_qdrant_query, mock_post):
        with patch("chatboton.agent.create_reranker_agent") as mock_create_reranker:
            mock_post.return_value.json.return_value = {"embedding": [0.1] * 768}
            mock_qdrant_query.return_value = json.dumps([{"payload": {"memory": "Memory 1"}}])
            mock_os_search.return_value = json.dumps([])
            
            # Mock Reranker failure
            mock_reranker = MagicMock()
            mock_create_reranker.return_value = mock_reranker
            mock_reranker.invoke.side_effect = Exception("Agent failed")
            
            result = search_long_term_memory.invoke({"query": "test query", "limit": 3})
            
            # Should return fallback results
            final_memories = json.loads(result)
            assert len(final_memories) == 1
            assert final_memories[0] == "Memory 1"

    def test_reranker_agent_creation(self):
        with patch("chatboton.agent.get_provider") as mock_get_provider:
            mock_provider = MagicMock()
            mock_get_provider.return_value = mock_provider
            
            agent = create_reranker_agent()
            assert agent is not None
            mock_provider.create_model.assert_called_once()
