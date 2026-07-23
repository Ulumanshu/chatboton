"""Provider tests — all four providers, no network calls (model classes mocked)."""

from unittest.mock import patch

import pytest

from chatboton.providers import (
    AnthropicProvider,
    AzureOpenAIProvider,
    OllamaProvider,
    OpenAIProvider,
    get_provider,
)


class TestOllamaProvider:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        provider = OllamaProvider()
        assert provider.model == "chatboton-heretic"
        assert provider.base_url == "http://localhost:11434"

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
        assert OllamaProvider().model == "qwen2.5:7b"

    def test_explicit_args_beat_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
        assert OllamaProvider(model="other").model == "other"

    def test_create_model(self):
        with patch("chatboton.providers.ChatOllama") as chat_ollama:
            model = OllamaProvider(model="m", base_url="http://x:1").create_model()
        chat_ollama.assert_called_once_with(
            model="m", base_url="http://x:1", temperature=0,
            num_ctx=16384, num_predict=2048,
        )
        assert model is chat_ollama.return_value


class TestOpenAIProvider:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIProvider()

    def test_create_model(self):
        with patch("chatboton.providers.ChatOpenAI") as chat_openai:
            model = OpenAIProvider(model="gpt-4o", api_key="sk-test").create_model()
        chat_openai.assert_called_once_with(model="gpt-4o", api_key="sk-test", temperature=0)
        assert model is chat_openai.return_value

    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        assert OpenAIProvider(api_key="sk-test").model == "gpt-4o-mini"


class TestAnthropicProvider:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicProvider()

    def test_create_model(self):
        with patch("chatboton.providers.ChatAnthropic") as chat_anthropic:
            model = AnthropicProvider(model="claude-haiku-4-5", api_key="sk-ant").create_model()
        chat_anthropic.assert_called_once_with(
            model="claude-haiku-4-5", api_key="sk-ant", temperature=0
        )
        assert model is chat_anthropic.return_value


class TestAzureOpenAIProvider:
    def test_requires_all_azure_settings(self, monkeypatch):
        for var in ("AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ValueError, match="AZURE_OPENAI_DEPLOYMENT"):
            AzureOpenAIProvider()
        with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
            AzureOpenAIProvider(deployment="d")
        with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
            AzureOpenAIProvider(deployment="d", endpoint="https://e")

    def test_create_model(self):
        with patch("chatboton.providers.AzureChatOpenAI") as azure_chat:
            provider = AzureOpenAIProvider(
                deployment="gpt-4o-mini", endpoint="https://res.openai.azure.com",
                api_key="key", api_version="2024-10-21",
            )
            model = provider.create_model()
        azure_chat.assert_called_once_with(
            azure_deployment="gpt-4o-mini",
            azure_endpoint="https://res.openai.azure.com",
            api_key="key", api_version="2024-10-21", temperature=0,
        )
        assert model is azure_chat.return_value


class TestGetProvider:
    def test_defaults_to_ollama(self, monkeypatch):
        monkeypatch.delenv("CHATBOTON_PROVIDER", raising=False)
        assert isinstance(get_provider(), OllamaProvider)

    def test_env_selection(self, monkeypatch):
        monkeypatch.setenv("CHATBOTON_PROVIDER", "openai")
        assert isinstance(get_provider(api_key="sk-test"), OpenAIProvider)

    def test_explicit_name_beats_env(self, monkeypatch):
        monkeypatch.setenv("CHATBOTON_PROVIDER", "openai")
        assert isinstance(get_provider("anthropic", api_key="sk-ant"), AnthropicProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider 'nope'"):
            get_provider("nope")
