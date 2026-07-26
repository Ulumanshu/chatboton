"""Relational tool: query the product inventory via SQL."""

from langchain_core.tools import tool
from chatboton.connectors.postgres import PostgresConnector

@tool
def query_postgres_inventory(sql: str) -> str:
    """Runs a SQL query against the gadget-store Postgres.

    The database has one table:
    products(id, name, category, price_eur, stock).

    Args:
        sql: A SQL statement (SELECT, INSERT, UPDATE, etc.).

    Returns:
        JSON list of row objects (for SELECT) or success status, or an error description.
    """
    return PostgresConnector(collection="products").query(sql)
