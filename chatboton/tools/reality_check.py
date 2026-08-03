"""Self-diagnostic tool: snapshots host + Ollama server state ("reality check").

The tool gathers the Ollama server memory usage, system-wide memory, disk and
CPU figures, and the provider-specific context-window token counters. Every
reading is persisted as a row in Postgres (table created on the fly via the
connector), and the previous reading is returned alongside the current one so
the agent can compare state over time. A short-term memory prefixed with
``SELF STATE:`` is also committed so the snapshot becomes searchable later.
"""

import json
import os
import platform
import time

import psutil
import requests
from langchain_core.tools import tool

from chatboton.connectors.postgres import PostgresConnector
from chatboton.context_state import get_current_context
from chatboton.providers import get_provider

REALITY_CHECK_TABLE = "reality_checks"

CREATE_TABLE_SQL = (
    f"CREATE TABLE IF NOT EXISTS {REALITY_CHECK_TABLE} ("
    "id SERIAL PRIMARY KEY, "
    "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "reading JSONB NOT NULL)"
)


def get_ollama_memory_usage(base_url=None) -> dict:
    """Returns memory usage of the models currently loaded in Ollama.

    Uses the ``/api/ps`` endpoint of the local Ollama server.

    Args:
        base_url: Ollama server URL; falls back to ``OLLAMA_BASE_URL``.

    Returns:
        dict: ``models`` (name, size bytes, VRAM bytes each) and totals,
            or an ``error`` key when the server is unreachable.
    """
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        resp = requests.get(f"{base_url}/api/ps", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except Exception as exc:
        return {"error": f"Ollama unreachable: {exc}"}
    return {
        "models": [
            {
                "name": m.get("name"),
                "size_bytes": m.get("size", 0),
                "size_vram_bytes": m.get("size_vram", 0),
            }
            for m in models
        ],
        "total_size_bytes": sum(m.get("size", 0) for m in models),
        "total_vram_bytes": sum(m.get("size_vram", 0) for m in models),
    }


def get_system_stats() -> dict:
    """Returns current host parameters via psutil.

    Covers total/used/free virtual memory, free disk space, CPU load,
    platform info and uptime.
    """
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return {
        "memory_total_bytes": memory.total,
        "memory_used_bytes": memory.used,
        "memory_free_bytes": memory.available,
        "memory_percent": memory.percent,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "disk_percent": disk.percent,
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "cpu_count": psutil.cpu_count(),
        "platform": platform.platform(),
        "uptime_seconds": int(time.time() - psutil.boot_time()),
    }


def get_context_token_counters(context_text: str = None) -> dict:
    """Returns provider-specific token counters for the current context.

    Args:
        context_text: The context to count tokens for; defaults to the
            conversation context recorded by the web app for this request.

    Returns:
        dict: Provider name and token count (0 when counting is impossible).
    """
    if context_text is None:
        context_text = get_current_context()
    provider_name = os.getenv("CHATBOTON_PROVIDER", "ollama")
    try:
        provider = get_provider(provider_name)
        tokens = provider.count_context_tokens(context_text)
    except Exception:
        tokens = 0
    return {"provider": provider_name, "context_tokens": tokens}


def _save_and_fetch_previous(reading: dict, connector=None) -> dict:
    """Persists the reading in Postgres and returns the previous one.

    Creates the ``reality_checks`` table if it does not exist, fetches the
    latest stored reading, then inserts the current one.
    """
    connector = connector or PostgresConnector(collection=REALITY_CHECK_TABLE)
    connector.query(CREATE_TABLE_SQL)
    previous_raw = connector.query(
        f"SELECT created_at, reading FROM {REALITY_CHECK_TABLE} "
        "ORDER BY id DESC LIMIT 1"
    )
    try:
        rows = json.loads(previous_raw)
        previous = rows[0] if isinstance(rows, list) and rows else None
    except (json.JSONDecodeError, TypeError):
        previous = {"error": previous_raw}
    payload = json.dumps(reading, default=str).replace("'", "''")
    connector.query(
        f"INSERT INTO {REALITY_CHECK_TABLE} (reading) VALUES ('{payload}')"
    )
    return previous


def _commit_self_state_memory(reading: dict) -> str:
    """Stores the reading as a ``SELF STATE:`` short-term memory."""
    from chatboton.tools.commit_short_term_memory import commit_short_term_memory
    note = "SELF STATE: " + json.dumps(reading, default=str)
    try:
        return commit_short_term_memory.func(note)
    except Exception as exc:
        return f"Short-term memory error: {exc}"


@tool
def reality_check() -> str:
    """Performs a self-diagnostic snapshot of the host and Ollama server.

    Reports Ollama server memory usage, total/used/free system memory, free
    disk space, CPU load, uptime and provider context token counters. The
    reading is stored in Postgres, and the previous reading is returned for
    comparison.

    Returns:
        JSON with ``current`` and ``previous`` readings, or an error string.
    """
    reading = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ollama": get_ollama_memory_usage(),
        "system": get_system_stats(),
        "context_tokens": get_context_token_counters(),
    }
    previous = _save_and_fetch_previous(reading)
    memory_status = _commit_self_state_memory(reading)
    return json.dumps(
        {"current": reading, "previous": previous, "self_state_memory": memory_status},
        default=str,
    )
