import json
import os
import psycopg
from .base import Connector

DEFAULT_DSN = "postgresql://chatboton:chatboton@localhost:55433/chatboton"

class PostgresConnector(Connector):
    def __init__(self, collection, dsn=None):
        self.collection = collection
        self.dsn = dsn or os.getenv("POSTGRES_DSN", DEFAULT_DSN)

    def query(self, sql: str) -> str:
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as conn:
                cursor = conn.execute(sql)
                if cursor.description:
                    columns = [c.name for c in cursor.description]
                    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    return json.dumps(rows, default=str)
                else:
                    return json.dumps({"status": "success", "rowcount": cursor.rowcount})
        except psycopg.Error as exc:
            return f"Postgres error: {exc}"

    def insert(self, sql: str) -> str:
        # For Postgres, insert is just an EXECUTE of an INSERT statement
        return self.query(sql)
