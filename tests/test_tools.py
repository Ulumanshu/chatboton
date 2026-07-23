"""Tool tests — one per database, all clients mocked (no docker required)."""

import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

from chatboton.tool_log import ToolInvocationLog
from chatboton.tools import default_tools
from chatboton.tools.chroma_tool import ChromaTool, ChromaConnector
from chatboton.tools.neo4j_tool import Neo4jTool, Neo4jConnector
from chatboton.tools.postgres_tool import PostgresTool, PostgresConnector
from chatboton.tools.qdrant_tool import QdrantTool, QdrantConnector
from chatboton.tools.opensearch_tool import OpenSearchTool, OpenSearchConnector


class TestPostgresTool:
    def test_rejects_non_select(self):
        assert "only SELECT" in PostgresTool(connector=PostgresConnector(dsn="x")).run("DELETE FROM products")

    def test_returns_rows_as_json(self):
        cursor = MagicMock()
        cursor.description = [MagicMock(name_attr="n"), MagicMock(name_attr="p")]
        cursor.description[0].name = "name"
        cursor.description[1].name = "price_eur"
        cursor.fetchall.return_value = [("Volta Powerbank 20k", Decimal("39.90"))]
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.return_value = cursor
        with patch("chatboton.tools.postgres_tool.psycopg.connect", return_value=connection) as connect:
            result = PostgresTool(connector=PostgresConnector(dsn="postgresql://u:p@h/db")).run("SELECT name, price_eur FROM products")
        connect.assert_called_once_with("postgresql://u:p@h/db", connect_timeout=5)
        assert json.loads(result) == [{"name": "Volta Powerbank 20k", "price_eur": "39.90"}]

    def test_as_tool_metadata(self):
        lc_tool = PostgresTool().as_tool()
        assert lc_tool.name == "query_postgres"
        assert "SELECT" in lc_tool.description


class TestNeo4jTool:
    def test_rejects_write_cypher(self):
        tool = Neo4jTool(connector=Neo4jConnector(uri="bolt://x", user="u", password="p"))
        for cypher in ("CREATE (n)", "MATCH (n) DETACH DELETE n", "merge (n)"):
            assert "read-only" in tool.run(cypher)

    def test_returns_records_as_json(self):
        record = MagicMock()
        record.data.return_value = {"customer": "Ada", "product": "Volta Powerbank 20k"}
        driver = MagicMock()
        driver.__enter__.return_value = driver
        driver.execute_query.return_value = ([record], None, None)
        with patch("chatboton.tools.neo4j_tool.GraphDatabase.driver", return_value=driver):
            result = Neo4jTool(connector=Neo4jConnector(uri="bolt://x", user="u", password="p")).run(
                "MATCH (c:Customer)-[:BOUGHT]->(p) RETURN c.name AS customer, p.name AS product"
            )
        assert json.loads(result) == [{"customer": "Ada", "product": "Volta Powerbank 20k"}]

    def test_as_tool_metadata(self):
        lc_tool = Neo4jTool().as_tool()
        assert lc_tool.name == "query_neo4j"
        assert "Cypher" in lc_tool.description


class TestChromaTool:
    def _tool_with_mocked_collection(self, collection):
        connector = ChromaConnector(host="h", port=1, collection="product_reviews")
        connector._get_collection = MagicMock(return_value=collection)
        return ChromaTool(connector=connector)

    def test_returns_hits_as_json(self):
        collection = MagicMock()
        collection.query.return_value = {
            "documents": [["Great for hiking trips."]],
            "metadatas": [[{"product": "Volta Powerbank 20k", "rating": 5}]],
            "distances": [[0.1234567]],
        }
        tool = self._tool_with_mocked_collection(collection)
        hits = json.loads(tool.run("battery for travel", n_results=1))
        collection.query.assert_called_once_with(query_texts=["battery for travel"], n_results=1)
        assert hits == [{
            "document": "Great for hiking trips.",
            "metadata": {"product": "Volta Powerbank 20k", "rating": 5},
            "distance": 0.1235,
        }]

    def test_connection_error_is_reported_not_raised(self):
        connector = ChromaConnector(host="h", port=1)
        connector._get_collection = MagicMock(side_effect=RuntimeError("connection refused"))
        tool = ChromaTool(connector=connector)
        assert tool.run("anything").startswith("Chroma error:")

    def test_as_tool_metadata(self):
        lc_tool = ChromaTool().as_tool()
        assert lc_tool.name == "search_reviews"
        assert "review" in lc_tool.description.lower()


class TestQdrantTool:
    def test_returns_hits_as_json(self):
        client = MagicMock()
        hit = MagicMock()
        hit.id = 1
        hit.payload = {"name": "Volta Powerbank 20k"}
        hit.score = 0.99
        client.query_points.return_value = MagicMock(points=[hit])
        
        connector = QdrantConnector(host="h", port=1, collection="c")
        connector._client = client
        tool = QdrantTool(connector=connector)
        
        # Test connector directly since as_tool() requires requests/ollama
        result = connector.query([0.1, 0.2, 0.3], limit=1)
        assert json.loads(result) == [{"id": 1, "payload": {"name": "Volta Powerbank 20k"}, "score": 0.99}]
        client.query_points.assert_called_once()

    def test_as_tool_metadata(self):
        lc_tool = QdrantTool().as_tool()
        assert lc_tool.name == "search_products_v2"
        assert "Qdrant" in lc_tool.description


class TestOpenSearchTool:
    def test_returns_hits_as_json(self):
        client = MagicMock()
        client.search.return_value = {"hits": {"hits": [{"_source": {"name": "Volta"}}]}}
        
        connector = OpenSearchConnector(host="h", port=1, index="i")
        connector._client = client
        tool = OpenSearchTool(connector=connector)
        
        result = connector.query({"query": {"match_all": {}}})
        assert json.loads(result) == [{"_source": {"name": "Volta"}}]
        client.search.assert_called_once()

    def test_as_tool_metadata(self):
        lc_tool = OpenSearchTool().as_tool()
        assert lc_tool.name == "full_text_search"
        assert "OpenSearch" in lc_tool.description


def test_default_tools_covers_every_database():
    names = {t.name for t in default_tools()}
    assert names == {"query_postgres", "query_neo4j", "search_reviews", "search_products_v2", "full_text_search"}


class TestToolInvocationLog:
    def test_record_entries_clear(self):
        log = ToolInvocationLog()
        assert log.entries() == []
        log.record("query_postgres", {"sql": "SELECT 1"}, "[]")
        entries = log.entries()
        assert len(entries) == 1
        assert entries[0]["tool"] == "query_postgres"
        assert entries[0]["args"] == {"sql": "SELECT 1"}
        assert entries[0]["result"] == "[]"
        assert entries[0]["timestamp"]
        log.clear()
        assert log.entries() == []

    def test_entries_returns_copy(self):
        log = ToolInvocationLog()
        log.record("t", {}, None)
        log.entries().clear()
        assert len(log.entries()) == 1
