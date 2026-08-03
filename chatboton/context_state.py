"""Shared snapshot of the current conversation context.

The web app updates this on every chat request so tools that run inside the
agent (e.g. ``reality_check``) can measure the real context — the system
prompt plus the whole message history — instead of an empty string.
"""

_current_context = ""


def set_current_context(text: str) -> None:
    """Stores the latest conversation context as a plain string."""
    global _current_context
    _current_context = text or ""


def get_current_context() -> str:
    """Returns the last stored conversation context (empty when none)."""
    return _current_context


def build_context_text(system_prompt: str, messages: list) -> str:
    """Flattens the system prompt and chat messages into one countable string.

    Args:
        system_prompt: The agent system prompt.
        messages: List of ``{"role": ..., "content": ...}`` dicts.

    Returns:
        str: ``role: content`` lines, one per message, prefixed by the prompt.
    """
    lines = [f"system: {system_prompt}"] if system_prompt else []
    for message in messages or []:
        role = message.get("role", "user") if isinstance(message, dict) else "user"
        content = message.get("content", "") if isinstance(message, dict) else str(message)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
