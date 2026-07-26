import json
import os
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from .base import Connector


class Neo4jConnector(Connector):
    def __init__(self, collection, uri=None, user=None, password=None):
        self.collection = collection
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:57688")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "chatboton_password")

    def query(self, cypher: str) -> str:
        try:
            with GraphDatabase.driver(self.uri, auth=(self.user, self.password)) as driver:
                records, _, _ = driver.execute_query(cypher)
            return json.dumps([record.data() for record in records], default=str)
        except (Neo4jError, ServiceUnavailable, OSError) as exc:
            return f"Neo4j error: {exc}"

    def insert(self, cypher: str) -> str:
        # For Neo4j, insert is often just a MERGE or CREATE cypher query
        return self.query(cypher)
