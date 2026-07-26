"""Graph tool: read-only Cypher against the purchase graph."""

from langchain_core.tools import tool
from chatboton.connectors.neo4j import Neo4jConnector

@tool
def query_purchase_graph(cypher: str) -> str:
    """Runs a Cypher query against the gadget-store purchase graph.

    The graph contains (:Customer {name, city})-[:BOUGHT {qty}]->
    (:Product {name, category}) nodes. Product names match the
    Postgres products table.

    Args:
        cypher: A Cypher query (MATCH, CREATE, etc.).

    Returns:
        JSON list of result records, or an error description.
    """
    return Neo4jConnector(collection="purchases").query(cypher)
