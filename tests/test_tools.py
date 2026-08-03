"""Tool tests — one per database, all clients mocked (no docker required)."""

import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

from chatboton.tool_log import ToolInvocationLog
from chatboton.tools import default_tools
from chatboton.tools.chroma_tool import search_customer_reviews
from chatboton.connectors.chroma import ChromaConnector
from chatboton.tools.neo4j_tool import query_purchase_graph
from chatboton.connectors.neo4j import Neo4jConnector
from chatboton.tools.postgres_tool import query_postgres_inventory
from chatboton.connectors.postgres import PostgresConnector
from chatboton.tools.qdrant_tool import search_product_catalog
from chatboton.tools.search_long_term_memory import search_long_term_memory
from chatboton.connectors.qdrant import QdrantConnector
from chatboton.tools.opensearch_tool import full_text_search_docs
from chatboton.connectors.opensearch import OpenSearchConnector


class TestPostgresTool:
    def test_supports_insert(self):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        cursor = MagicMock()
        cursor.description = None
        cursor.rowcount = 1
        connection.execute.return_value = cursor
        with patch("chatboton.connectors.postgres.psycopg.connect", return_value=connection):
            result = PostgresConnector(collection="products", dsn="postgresql://u:p@h/db").query("INSERT INTO products DEFAULT VALUES")
        assert json.loads(result) == {"status": "success", "rowcount": 1}

    def test_returns_rows_as_json(self):
        cursor = MagicMock()
        cursor.description = [MagicMock(name_attr="n"), MagicMock(name_attr="p")]
        cursor.description[0].name = "name"
        cursor.description[1].name = "price_eur"
        cursor.fetchall.return_value = [("Volta Powerbank 20k", Decimal("39.90"))]
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.return_value = cursor
        with patch("chatboton.connectors.postgres.psycopg.connect", return_value=connection) as connect:
            result = PostgresConnector(collection="products", dsn="postgresql://u:p@h/db").query("SELECT name, price_eur FROM products")
        connect.assert_called_once_with("postgresql://u:p@h/db", connect_timeout=5)
        assert json.loads(result) == [{"name": "Volta Powerbank 20k", "price_eur": "39.90"}]

    def test_tool_metadata(self):
        lc_tool = query_postgres_inventory
        assert lc_tool.name == "query_postgres_inventory"
        assert "SELECT" in lc_tool.description


class TestNeo4jTool:
    def test_supports_write_cypher(self):
        driver = MagicMock()
        driver.__enter__.return_value = driver
        driver.execute_query.return_value = ([], None, None)
        with patch("chatboton.connectors.neo4j.GraphDatabase.driver", return_value=driver):
            result = Neo4jConnector(collection="purchases", uri="bolt://x", user="u", password="p").query("CREATE (n)")
        assert json.loads(result) == []

    def test_returns_records_as_json(self):
        record = MagicMock()
        record.data.return_value = {"customer": "Ada", "product": "Volta Powerbank 20k"}
        driver = MagicMock()
        driver.__enter__.return_value = driver
        driver.execute_query.return_value = ([record], None, None)
        with patch("chatboton.connectors.neo4j.GraphDatabase.driver", return_value=driver):
            result = Neo4jConnector(collection="purchases", uri="bolt://x", user="u", password="p").query(
                "MATCH (c:Customer)-[:BOUGHT]->(p) RETURN c.name AS customer, p.name AS product"
            )
        assert json.loads(result) == [{"customer": "Ada", "product": "Volta Powerbank 20k"}]

    def test_tool_metadata(self):
        lc_tool = query_purchase_graph
        assert lc_tool.name == "query_purchase_graph"
        assert "Cypher" in lc_tool.description


class TestChromaTool:
    def test_returns_hits_as_json(self):
        collection = MagicMock()
        collection.query.return_value = {
            "documents": [["Great for hiking trips."]],
            "metadatas": [[{"product": "Volta Powerbank 20k", "rating": 5}]],
            "distances": [[0.1234567]],
        }
        connector = ChromaConnector(collection="product_reviews", host="h", port=1)
        with patch.object(ChromaConnector, "_get_collection", return_value=collection):
            result = connector.query("battery for travel", n_results=1)
        
        collection.query.assert_called_once_with(query_texts=["battery for travel"], n_results=1)
        hits = json.loads(result)
        assert hits == [{
            "document": "Great for hiking trips.",
            "metadata": {"product": "Volta Powerbank 20k", "rating": 5},
            "distance": 0.1235,
        }]

    def test_connection_error_is_reported_not_raised(self):
        connector = ChromaConnector(collection="product_reviews", host="h", port=1)
        with patch.object(ChromaConnector, "_get_collection", side_effect=RuntimeError("connection refused")):
            result = connector.query("anything")
        assert result.startswith("Chroma error:")

    def test_tool_metadata(self):
        lc_tool = search_customer_reviews
        assert lc_tool.name == "search_customer_reviews"
        assert "review" in lc_tool.description.lower()


class TestQdrantTool:
    def test_returns_hits_as_json(self):
        client = MagicMock()
        hit = MagicMock()
        hit.id = 1
        hit.payload = {"name": "Volta Powerbank 20k"}
        hit.score = 0.99
        client.query_points.return_value = MagicMock(points=[hit])
        
        connector = QdrantConnector(collection="c", host="h", port=1)
        connector._client = client
        
        result = connector.query([0.1, 0.2, 0.3], limit=1)
        assert json.loads(result) == [{"id": 1, "payload": {"name": "Volta Powerbank 20k"}, "score": 0.99}]
        client.query_points.assert_called_once()

    def test_tool_metadata(self):
        lc_tool = search_product_catalog
        assert lc_tool.name == "search_product_catalog"
        assert "Qdrant" in lc_tool.description

class TestSearchLongTermMemoryTool:
    def test_tool_metadata(self):
        lc_tool = search_long_term_memory
        assert lc_tool.name == "search_long_term_memory"
        assert "long-term" in lc_tool.description.lower()


class TestOpenSearchTool:
    def test_returns_hits_as_json(self):
        client = MagicMock()
        client.search.return_value = {"hits": {"hits": [{"_source": {"name": "Volta"}}]}}
        
        connector = OpenSearchConnector(collection="products", host="h", port=1)
        connector._client = client
        
        result = connector.query({"query": {"match_all": {}}})
        assert json.loads(result) == [{"_source": {"name": "Volta"}}]
        client.search.assert_called_once()

    def test_tool_metadata(self):
        lc_tool = full_text_search_docs
        assert lc_tool.name == "full_text_search_docs"
        assert "OpenSearch" in lc_tool.description


def test_default_tools_covers_every_database():
    names = {t.name for t in default_tools()}
    assert names == {
        "search_long_term_memory",
        "reality_check"
    }


class TestToolInvocationLog:
    def test_record_entries_clear(self):
        log = ToolInvocationLog()
        assert log.entries() == []
        log.record("query_postgres_inventory", {"sql": "SELECT 1"}, "[]")
        entries = log.entries()
        assert len(entries) == 1
        assert entries[0]["tool"] == "query_postgres_inventory"
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
