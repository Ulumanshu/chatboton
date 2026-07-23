"""Swappable LLM provider classes.

Every provider builds a LangChain ``BaseChatModel``, so the agent works
identically on Ollama (default, local, free), OpenAI, Anthropic, or Azure
OpenAI. Selection order: explicit ``get_provider(name)`` argument, then the
``CHATBOTON_PROVIDER`` environment variable, then ``"ollama"``.

Constructor arguments always win over environment variables, which win over
the built-in defaults — so tests and callers can configure providers without
touching the environment.
"""

import os
from abc import ABC, abstractmethod

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

# Heretic-abliterated Qwen2.5-3B rebuilt with a tool-aware chat template —
# see ollama/Modelfile for the build command.
DEFAULT_OLLAMA_MODEL = "chatboton-heretic"


class BaseProvider(ABC):
    """Abstract provider: one ``create_model()`` returning a chat model."""

    @abstractmethod
    def create_model(self):
        """Returns a LangChain ``BaseChatModel`` for this provider."""


class OllamaProvider(BaseProvider):
    """Local Ollama server running a Heretic-abliterated model by default.

    Args:
        model: Ollama model tag; falls back to ``OLLAMA_MODEL``.
        base_url: Ollama server URL; falls back to ``OLLAMA_BASE_URL``.
        temperature: Sampling temperature.
        num_ctx: Context window — Ollama's 4096 default silently truncates
            long tool-calling conversations, so we raise it.
        num_predict: Hard cap on generated tokens per response.
    """

    def __init__(self, model=None, base_url=None, temperature=0,
                 num_ctx=None, num_predict=None):
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.temperature = temperature
        self.num_ctx = num_ctx or int(os.getenv("OLLAMA_NUM_CTX", "16384"))
        self.num_predict = num_predict or int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))

    def create_model(self):
        return ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
        )


class OpenAIProvider(BaseProvider):
    """OpenAI chat completions.

    Args:
        model: Model name; falls back to ``OPENAI_MODEL``, then gpt-4o-mini.
        api_key: Falls back to ``OPENAI_API_KEY``; required.
        temperature: Sampling temperature.
    """

    def __init__(self, model=None, api_key=None, temperature=0):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")

    def create_model(self):
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
        )


class AnthropicProvider(BaseProvider):
    """Anthropic Claude models.

    Args:
        model: Model name; falls back to ``ANTHROPIC_MODEL``, then
            claude-haiku-4-5.
        api_key: Falls back to ``ANTHROPIC_API_KEY``; required.
        temperature: Sampling temperature.
    """

    def __init__(self, model=None, api_key=None, temperature=0):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = temperature
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")

    def create_model(self):
        return ChatAnthropic(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
        )


class AzureOpenAIProvider(BaseProvider):
    """Azure-hosted OpenAI deployment.

    Args:
        deployment: Azure deployment name; falls back to
            ``AZURE_OPENAI_DEPLOYMENT``; required.
        endpoint: Resource endpoint URL; falls back to
            ``AZURE_OPENAI_ENDPOINT``; required.
        api_key: Falls back to ``AZURE_OPENAI_API_KEY``; required.
        api_version: Falls back to ``AZURE_OPENAI_API_VERSION``.
        temperature: Sampling temperature.
    """

    def __init__(self, deployment=None, endpoint=None, api_key=None,
                 api_version=None, temperature=0):
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        self.temperature = temperature
        for attr in ("deployment", "endpoint", "api_key"):
            if not getattr(self, attr):
                raise ValueError(f"AZURE_OPENAI_{attr.upper()} is not set")

    def create_model(self):
        return AzureChatOpenAI(
            azure_deployment=self.deployment,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            temperature=self.temperature,
        )


PROVIDERS = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "azure_openai": AzureOpenAIProvider,
}


def get_provider(name=None, **kwargs) -> BaseProvider:
    """Builds the provider selected by argument, env, or default.

    Args:
        name: Provider key; falls back to ``CHATBOTON_PROVIDER``, then
            ``"ollama"``.
        **kwargs: Passed to the provider constructor.

    Returns:
        BaseProvider: Configured provider instance.

    Raises:
        ValueError: On an unknown provider name.
    """
    name = (name or os.getenv("CHATBOTON_PROVIDER", "ollama")).lower()
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {name!r}; expected one of {sorted(PROVIDERS)}"
        )
    return PROVIDERS[name](**kwargs)
