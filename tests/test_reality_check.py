"""Reality-check tool tests — Ollama, psutil, Postgres and providers mocked."""

import json
from unittest.mock import MagicMock, patch

from chatboton.tools import default_tools
from chatboton.tools.reality_check import (
    CREATE_TABLE_SQL,
    get_context_token_counters,
    get_ollama_memory_usage,
    get_system_stats,
    reality_check,
    _commit_self_state_memory,
    _save_and_fetch_previous,
)
from chatboton.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
)


class TestOllamaMemoryHelper:
    def test_reports_loaded_models_and_totals(self):
        response = MagicMock()
        response.json.return_value = {"models": [
            {"name": "chatboton-heretic", "size": 100, "size_vram": 60},
            {"name": "nomic-embed-text", "size": 50, "size_vram": 40},
        ]}
        with patch("chatboton.tools.reality_check.requests.get", return_value=response) as get:
            result = get_ollama_memory_usage(base_url="http://ollama:11434")
        get.assert_called_once_with("http://ollama:11434/api/ps", timeout=5)
        assert result["total_size_bytes"] == 150
        assert result["total_vram_bytes"] == 100
        assert result["models"][0]["name"] == "chatboton-heretic"

    def test_returns_error_when_server_unreachable(self):
        with patch("chatboton.tools.reality_check.requests.get", side_effect=OSError("down")):
            result = get_ollama_memory_usage()
        assert "error" in result and "down" in result["error"]


class TestSystemStatsHelper:
    def test_reports_memory_disk_and_cpu(self):
        memory = MagicMock(total=16, used=8, available=8, percent=50.0)
        disk = MagicMock(total=100, free=40, percent=60.0)
        with patch("chatboton.tools.reality_check.psutil") as psutil_mock:
            psutil_mock.virtual_memory.return_value = memory
            psutil_mock.disk_usage.return_value = disk
            psutil_mock.cpu_percent.return_value = 12.5
            psutil_mock.cpu_count.return_value = 8
            psutil_mock.boot_time.return_value = 0
            result = get_system_stats()
        assert result["memory_total_bytes"] == 16
        assert result["memory_free_bytes"] == 8
        assert result["disk_free_bytes"] == 40
        assert result["cpu_percent"] == 12.5
        assert result["uptime_seconds"] > 0


class TestContextTokenCounters:
    def test_uses_selected_provider(self):
        provider = MagicMock()
        provider.count_context_tokens.return_value = 42
        with patch("chatboton.tools.reality_check.get_provider", return_value=provider):
            result = get_context_token_counters("hello")
        provider.count_context_tokens.assert_called_once_with("hello")
        assert result == {"provider": "ollama", "context_tokens": 42}

    def test_defaults_to_recorded_conversation_context(self):
        from chatboton.context_state import set_current_context
        provider = MagicMock()
        provider.count_context_tokens.return_value = 9
        set_current_context("system: prompt\nuser: hello")
        try:
            with patch("chatboton.tools.reality_check.get_provider", return_value=provider):
                result = get_context_token_counters()
        finally:
            set_current_context("")
        provider.count_context_tokens.assert_called_once_with("system: prompt\nuser: hello")
        assert result["context_tokens"] == 9

    def test_returns_zero_when_provider_fails(self):
        with patch("chatboton.tools.reality_check.get_provider", side_effect=ValueError("no key")):
            result = get_context_token_counters()
        assert result["context_tokens"] == 0


class TestPostgresPersistence:
    def test_creates_table_fetches_previous_and_inserts(self):
        connector = MagicMock()
        previous = [{"created_at": "2026-08-02", "reading": {"cpu": 1}}]
        connector.query.side_effect = [
            json.dumps({"status": "success", "rowcount": 0}),
            json.dumps(previous),
            json.dumps({"status": "success", "rowcount": 1}),
        ]
        result = _save_and_fetch_previous({"cpu": 2}, connector=connector)
        assert connector.query.call_args_list[0].args[0] == CREATE_TABLE_SQL
        assert "SELECT" in connector.query.call_args_list[1].args[0]
        assert "INSERT INTO reality_checks" in connector.query.call_args_list[2].args[0]
        assert result == previous[0]

    def test_returns_none_when_table_empty(self):
        connector = MagicMock()
        connector.query.side_effect = ["{}", "[]", "{}"]
        assert _save_and_fetch_previous({"cpu": 2}, connector=connector) is None

    def test_wraps_connector_errors(self):
        connector = MagicMock()
        connector.query.side_effect = ["{}", "Postgres error: boom", "{}"]
        result = _save_and_fetch_previous({"cpu": 2}, connector=connector)
        assert result == {"error": "Postgres error: boom"}


class TestSelfStateMemory:
    def test_commits_prefixed_short_term_memory(self):
        with patch("chatboton.tools.commit_short_term_memory.commit_short_term_memory") as tool_mock:
            tool_mock.func.return_value = "ok"
            result = _commit_self_state_memory({"cpu": 1})
        note = tool_mock.func.call_args.args[0]
        assert note.startswith("SELF STATE: ")
        assert '"cpu": 1' in note
        assert result == "ok"

    def test_reports_memory_errors_without_raising(self):
        with patch("chatboton.tools.commit_short_term_memory.commit_short_term_memory") as tool_mock:
            tool_mock.func.side_effect = RuntimeError("qdrant down")
            result = _commit_self_state_memory({"cpu": 1})
        assert "qdrant down" in result


class TestRealityCheckTool:
    def test_returns_current_and_previous_readings(self):
        with patch("chatboton.tools.reality_check.get_ollama_memory_usage", return_value={"total_size_bytes": 1}), \
             patch("chatboton.tools.reality_check.get_system_stats", return_value={"memory_total_bytes": 2}), \
             patch("chatboton.tools.reality_check.get_context_token_counters", return_value={"provider": "ollama", "context_tokens": 0}), \
             patch("chatboton.tools.reality_check._save_and_fetch_previous", return_value={"reading": {"old": True}}) as save, \
             patch("chatboton.tools.reality_check._commit_self_state_memory", return_value="stored") as memory:
            result = json.loads(reality_check.func())
        assert result["current"]["ollama"] == {"total_size_bytes": 1}
        assert result["current"]["system"] == {"memory_total_bytes": 2}
        assert result["previous"] == {"reading": {"old": True}}
        assert result["self_state_memory"] == "stored"
        save.assert_called_once()
        memory.assert_called_once()

    def test_tool_metadata_and_registration(self):
        assert reality_check.name == "reality_check"
        assert "self-diagnostic" in reality_check.description.lower()
        assert any(t.name == "reality_check" for t in default_tools())


class TestProviderTokenCounters:
    def test_ollama_counts_via_server_prompt_eval(self):
        response = MagicMock()
        response.json.return_value = {"prompt_eval_count": 7}
        provider = OllamaProvider(model="chatboton-heretic", base_url="http://ollama:11434")
        with patch("chatboton.providers.requests.post", return_value=response) as post:
            assert provider.count_context_tokens("hi") == 7
        args, kwargs = post.call_args
        assert args[0] == "http://ollama:11434/api/generate"
        assert kwargs["json"]["model"] == "chatboton-heretic"
        assert kwargs["json"]["prompt"] == "hi"
        assert kwargs["json"]["options"] == {"num_predict": 0}

    def test_ollama_returns_zero_for_empty_context(self):
        provider = OllamaProvider(model="chatboton-heretic")
        with patch("chatboton.providers.requests.post") as post:
            assert provider.count_context_tokens("") == 0
        post.assert_not_called()

    def test_ollama_falls_back_to_langchain_when_server_fails(self):
        model = MagicMock()
        model.get_num_tokens.return_value = 5
        provider = OllamaProvider(model="chatboton-heretic")
        with patch("chatboton.providers.requests.post", side_effect=OSError("down")), \
             patch.object(OllamaProvider, "create_model", return_value=model):
            assert provider.count_context_tokens("hi") == 5
        model.get_num_tokens.assert_called_once_with("hi")

    def test_openai_returns_zero_on_failure(self):
        provider = OpenAIProvider(model="custom-model", api_key="k")
        with patch.object(OpenAIProvider, "create_model", side_effect=RuntimeError("boom")):
            assert provider.count_context_tokens("hi") == 0

    def test_anthropic_returns_zero_on_failure(self):
        model = MagicMock()
        model.get_num_tokens.side_effect = RuntimeError("no api")
        provider = AnthropicProvider(model="custom-claude", api_key="k")
        with patch.object(AnthropicProvider, "create_model", return_value=model):
            assert provider.count_context_tokens("hi") == 0
