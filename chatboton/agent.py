"""LangChain tool-calling agent wired to the three database tools."""

from langchain.agents import create_agent

from .providers import get_provider
from .tools import default_tools

SYSTEM_PROMPT = (
    "You are Chatboton, a user's assistant.\n"
    "You have a helpful and sarcastic tech-oriented personality, your user is a developer. \n"
    "Your purpose is to help the user and maintain a record of your interactions.\n"
    "You MUST commit every user message to memory using the available tools.\n"
    "Available tools:\n"
    "- commit_short_term_memory: Saves the current user request into short-term memory.\n"
    "- search_long_term_memory: Searches through long-term memory for relevant past information.\n"
    "Always use commit_short_term_memory for every user input to ensure it is remembered."
)


def create_chatboton_agent(provider=None, tools=None):
    """Builds the chat agent.

    Args:
        provider: Optional ``BaseProvider``; defaults to the env-selected
            provider (Ollama with the heretic model unless overridden).
        tools: Optional tool list; defaults to one tool per database.

    Returns:
        Runnable: LangChain agent invocable with ``{"messages": [...]}``.
    """
    provider = provider or get_provider()
    return create_agent(
        model=provider.create_model(),
        tools=tools if tools is not None else default_tools(),
        system_prompt=SYSTEM_PROMPT,
    )
