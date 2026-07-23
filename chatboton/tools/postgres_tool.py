"""Sample relational tool: read-only SQL against the demo Postgres."""

import json
import os

import psycopg
from .base import Connector

DEFAULT_DSN = "postgresql://chatboton:chatboton@localhost:55433/chatboton"


class PostgresConnector(Connector):
    def __init__(self, dsn=None):
        self.dsn = dsn or os.getenv("POSTGRES_DSN", DEFAULT_DSN)

    def query(self, sql: str) -> str:
        if not sql.strip().lower().startswith("select"):
            return "Error: only SELECT queries are allowed."
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as conn:
                cursor = conn.execute(sql)
                columns = [c.name for c in cursor.description]
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return json.dumps(rows, default=str)
        except psycopg.Error as exc:
            return f"Postgres error: {exc}"


class PostgresTool:
    """Runs SELECT queries against the seeded ``products`` table.

    Args:
        dsn: Postgres connection string; falls back to ``POSTGRES_DSN``, then
            the docker-compose default.
    """

    def __init__(self, connector: PostgresConnector = None):
        self.connector = connector or PostgresConnector()

    def run(self, sql: str) -> str:
        return self.connector.query(sql)

    def as_tool(self):
        from langchain_core.tools import tool

        @tool
        def query_postgres(sql: str) -> str:
            """Runs a read-only SQL SELECT against the gadget-store Postgres.

            The database has one table:
            products(id, name, category, price_eur, stock).

            Args:
                sql: A single SELECT statement.

            Returns:
                JSON list of row objects, or an error description.
            """
            return self.connector.query(sql)

        return query_postgres
