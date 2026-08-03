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
    "- reality_check: Runs a self-diagnostic snapshot (Ollama server memory usage, system memory,\n"
    "  free disk space, CPU load, provider context token counters), stores it in Postgres and\n"
    "  returns the current and previous readings. Use it when asked about your own state,\n"
    "  the host machine, or resource usage.\n"
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
    "- object: (string) who the memory is about (e.g. 'User').\n"
    "- subject: (string) what the memory is about, one short phrase.\n"
    "- sentiment: (string) 'positive', 'negative' or 'neutral'.\n"
    "- topics: (list of strings) topic tags, lowercase, short.\n"
    "- technologies: (list of strings) technologies mentioned in the message.\n"
    "- tags: (list of strings) any additional logical tags (activity type, domain, intent).\n"
    "- memory: (string) the clear, concise memory statement based on memory content facts\n"
    " (you can infer facts but then describe the inference logic in the memory) do not make stuff up.\n"
    " The memory must be a human-readable note with real facts from the user message,\n"
    " formatted in a way friendly for semantic search and with no fact distortion.\n"
    "If include is false, omit or leave the other fields empty.\n"
    "Example output:\n"
    '{"include": true, "object": "User", "subject": "React project work",\n'
    ' "sentiment": "positive", "topics": ["web development", "frontend"],\n'
    ' "technologies": ["React"], "tags": ["project", "work"],\n'
    ' "memory": "User is working on a React project and enjoys frontend development."}\n'
    "Output ONLY the JSON object."
)


def _as_str_list(value) -> list:
    """Coerces an LLM-produced value into a clean list of strings."""
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def extract_memory_metadata(data: dict) -> dict:
    """Extracts structured metadata tags from the transformer JSON output."""
    return {
        "object": str(data.get("object", "") or "").strip(),
        "subject": str(data.get("subject", "") or "").strip(),
        "sentiment": str(data.get("sentiment", "") or "").strip().lower(),
        "topics": _as_str_list(data.get("topics")),
        "technologies": _as_str_list(data.get("technologies")),
        "tags": _as_str_list(data.get("tags")),
    }


def build_memory_note(data: dict) -> str:
    """Builds a human-readable, semantic-search-friendly memory note.

    Example:
        Memory about User — React project work (positive).
        User is working on a React project and enjoys frontend development.
        Topics: web development, frontend. Technologies: React. Tags: project, work.
    """
    metadata = extract_memory_metadata(data)
    memory = str(data.get("memory", "") or "").strip()
    header = ""
    if metadata["object"] or metadata["subject"]:
        header = f"Memory about {metadata['object'] or 'User'}"
        if metadata["subject"]:
            header += f" — {metadata['subject']}"
        if metadata["sentiment"]:
            header += f" ({metadata['sentiment']})"
        header += "."
    lines = [line for line in (header, memory) if line]
    tag_bits = []
    if metadata["topics"]:
        tag_bits.append("Topics: " + ", ".join(metadata["topics"]) + ".")
    if metadata["technologies"]:
        tag_bits.append("Technologies: " + ", ".join(metadata["technologies"]) + ".")
    if metadata["tags"]:
        tag_bits.append("Tags: " + ", ".join(metadata["tags"]) + ".")
    if tag_bits:
        lines.append(" ".join(tag_bits))
    return "\n".join(lines)

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
