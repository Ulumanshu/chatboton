"""Integration tests against the real docker compose databases.

Skipped automatically when a database is unreachable. Run the stack first:
    docker compose up -d && python scripts/seed.py
"""

import json
import socket

import pytest

from chatboton.tools.chroma_tool import ChromaTool
from chatboton.tools.neo4j_tool import Neo4jTool
from chatboton.tools.postgres_tool import PostgresTool
from chatboton.tools.qdrant_tool import QdrantTool, QdrantConnector
from chatboton.tools.opensearch_tool import OpenSearchTool, OpenSearchConnector


def _port_open(host, port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


requires_postgres = pytest.mark.skipif(
    not _port_open("localhost", 55433), reason="postgres not running (docker compose up -d)"
)
requires_neo4j = pytest.mark.skipif(
    not _port_open("localhost", 57688), reason="neo4j not running (docker compose up -d)"
)
requires_chroma = pytest.mark.skipif(
    not (_port_open("localhost", 58001) and _port_open("localhost", 11434)),
    reason="chroma or ollama not running",
)
requires_qdrant = pytest.mark.skipif(
    not (_port_open("localhost", 58333) and _port_open("localhost", 11434)),
    reason="qdrant or ollama not running",
)
requires_opensearch = pytest.mark.skipif(
    not _port_open("localhost", 59200),
    reason="opensearch not running",
)


@requires_postgres
def test_postgres_tool_reads_seeded_products():
    rows = json.loads(PostgresTool().run("SELECT name, category FROM products ORDER BY id"))
    assert {"name": "Volta Powerbank 20k", "category": "power"} in rows
    assert len(rows) == 6


@requires_neo4j
def test_neo4j_tool_reads_seeded_purchases():
    rows = json.loads(Neo4jTool().run(
        "MATCH (c:Customer)-[b:BOUGHT]->(p:Product) "
        "RETURN c.name AS customer, p.name AS product, b.qty AS qty ORDER BY customer"
    ))
    assert len(rows) >= 7
    assert any(r["customer"] == "Ada" and r["product"] == "Volta Powerbank 20k" for r in rows)


@requires_chroma
def test_chroma_tool_finds_semantically_close_review():
    hits = json.loads(ChromaTool().run("portable battery for outdoor trips", n_results=2))
    assert len(hits) == 2
    assert any(h["metadata"]["product"] == "Volta Powerbank 20k" for h in hits)


@requires_qdrant
def test_qdrant_tool_finds_seeded_products():
    import requests
    ollama_url = "http://localhost:11434"
    resp = requests.post(
        f"{ollama_url}/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": "portable power"}
    )
    vector = resp.json()["embedding"]
    hits = json.loads(QdrantConnector().query(vector, limit=1))
    assert len(hits) == 1
    assert hits[0]["payload"]["name"] == "Volta Powerbank 20k"


@requires_opensearch
def test_opensearch_tool_finds_seeded_products():
    body = {"query": {"match": {"name": "Volta"}}}
    hits = json.loads(OpenSearchConnector().query(body))
    assert len(hits) >= 1
    assert hits[0]["_source"]["name"] == "Volta Powerbank 20k"
