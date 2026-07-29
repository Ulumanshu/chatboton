"""LangChain tool-calling agent wired to the database tools."""

from langchain.agents import create_agent

from .providers import get_provider
SYSTEM_PROMPT = (
    "You are Chatboton, a user's assistant.\n"
    "You have a helpful and sarcastic tech-oriented personality, your user is a developer. \n"
    "Your purpose is to help the user and maintain a record of your interactions.\n"
    "CRITICAL MANDATORY FLOW:\n"
    "1. For EVERY user message, you MUST FIRST call 'search_long_term_memory' to retrieve relevant past context.\n"
    "2. Only after this tool call, you may proceed with other tools or provide your response.\n"
    "Available tools:\n"
    "- search_long_term_memory: Searches through long-term memory for relevant past information.\n"
    "Do not mention these background memory steps to the user. Just execute the tool and provide your response."
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
    from .tools import default_tools
    provider = provider or get_provider()
    return create_agent(
        model=provider.create_model(),
        tools=tools if tools is not None else default_tools(),
        system_prompt=SYSTEM_PROMPT,
    )


import re
import json

MEMORY_TRANSFORMER_PROMPT = (
    "You are a Memory Transformer Agent.\n"
    "Your job is to analyze short-term memories and transform them into long-term memories.\n"
    "Extract and summarize everything that can help help user better in the future:\n"
    "- User identity and preferences\n"
    "- Projects and jobs they are working on\n"
    "- Interests and reactions\n"
    "- Technologies they use or are interested in\n"
    "Format the output as a JSON object with the following fields:\n"
    "- include: (boolean) whether this interaction contains information worth remembering for the long term.\n"
    "- memory: (string) the clear, concise memory statement based on memory content facts\n"
    " (you can infer facts but then describe the inference logic in the memory) do not make stuff up\n"
    " if include is true, otherwise omit or leave empty.\n"
    " format the memory in following fashion:\n"
    " Object: who the memory is about\n"
    " Subject: What the memory is about\n"
    " Sentiment: is it positive or negative\n"
    " Topics: list of topic tags\n"
    " Technologies: mentioned technologies\n"
    " chatboton_formatted_memmory: whole memory with real facts from user message formatted in a way friendly for semantic search and also no fact distortion\n"
    "Output ONLY the JSON object."
)

def parse_json_from_llm(text: str) -> dict:
    """Extracts and parses JSON from a string that might contain markdown blocks."""
    # Remove markdown code blocks
    text = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to find anything between { and }
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raise

def create_memory_transformer_agent(provider=None):
    """Builds the memory transformer agent."""
    provider = provider or get_provider()
    return create_agent(
        model=provider.create_model(),
        tools=[],
        system_prompt=MEMORY_TRANSFORMER_PROMPT,
    )


RERANKER_PROMPT = (
    "You are a Reranker Agent.\n"
    "Your job is to re-rank a list of search results based on their relevance to a user query.\n"
    "You will be provided with a user query and a list of memories (search results).\n"
    "Evaluate each memory and determine how relevant it is to the query.\n"
    "Return the re-ranked list of memories, starting with the most relevant.\n"
    "Output ONLY the re-ranked memories as a JSON list of strings, with no additional explanation."
)

def create_reranker_agent(provider=None):
    """Builds the reranker agent."""
    provider = provider or get_provider()
    return create_agent(
        model=provider.create_model(),
        tools=[],
        system_prompt=RERANKER_PROMPT,
    )
