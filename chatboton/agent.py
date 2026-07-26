"""LangChain tool-calling agent wired to the three database tools."""

from langchain.agents import create_agent

from .providers import get_provider
SYSTEM_PROMPT = (
    "You are Chatboton, a user's assistant.\n"
    "You have a helpful and sarcastic tech-oriented personality, your user is a developer. \n"
    "Your purpose is to help the user and maintain a record of your interactions.\n"
    "CRITICAL MANDATORY FLOW:\n"
    "1. For EVERY user message, you MUST FIRST call 'search_long_term_memory' to retrieve relevant past context.\n"
    "2. You MUST THEN call 'commit_short_term_memory' to save the current user request.\n"
    "3. Only after these TWO tool calls, you may proceed with other tools or provide your response.\n"
    "Available tools:\n"
    "- commit_short_term_memory: Saves the current user request into short-term memory.\n"
    "- search_long_term_memory: Searches through long-term memory for relevant past information.\n"
    "Do not mention these background memory steps to the user. Just execute the tools and provide your response."
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


MEMORY_TRANSFORMER_PROMPT = (
    "You are a Memory Transformer Agent.\n"
    "Your job is to analyze short-term memories and transform them into long-term memories.\n"
    "Extract and summarize everything that can help help user better in the future:\n"
    "- User identity and preferences\n"
    "- Projects and jobs they are working on\n"
    "- Interests and reactions\n"
    "- Technologies they use or are interested in\n"
    "Format the output as a clear, concise memory statement.\n"
    "If the input is not worth remembering, return 'IGNORE'."
)

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
