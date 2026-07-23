"""Sample graph tool: read-only Cypher against the demo Neo4j."""

import json
import os
import re

from .base import Connector
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

WRITE_KEYWORDS = re.compile(
    r"\b(create|merge|delete|detach|set|remove|drop)\b", re.IGNORECASE
)


class Neo4jConnector(Connector):
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:57688")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "chatboton_password")

    def query(self, cypher: str) -> str:
        if WRITE_KEYWORDS.search(cypher):
            return "Error: only read-only Cypher (MATCH/RETURN) is allowed."
        try:
            with GraphDatabase.driver(self.uri, auth=(self.user, self.password)) as driver:
                records, _, _ = driver.execute_query(cypher)
            return json.dumps([record.data() for record in records], default=str)
        except (Neo4jError, ServiceUnavailable, OSError) as exc:
            return f"Neo4j error: {exc}"


class Neo4jTool:
    """Runs read-only Cypher against the seeded purchase graph.

    Args:
        uri: Bolt URI; falls back to ``NEO4J_URI``.
        user: Username; falls back to ``NEO4J_USER``.
        password: Password; falls back to ``NEO4J_PASSWORD``.
    """

    def __init__(self, connector: Neo4jConnector = None):
        self.connector = connector or Neo4jConnector()

    def run(self, cypher: str) -> str:
        return self.connector.query(cypher)

    def as_tool(self):
        from langchain_core.tools import tool

        @tool
        def query_neo4j(cypher: str) -> str:
            """Runs read-only Cypher against the gadget-store purchase graph.

            The graph contains (:Customer {name, city})-[:BOUGHT {qty}]->
            (:Product {name, category}) nodes. Product names match the
            Postgres products table.

            Args:
                cypher: A read-only Cypher query (MATCH ... RETURN ...).

            Returns:
                JSON list of result records, or an error description.
            """
            return self.connector.query(cypher)

        return query_neo4j
