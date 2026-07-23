"""In-memory log of agent tool invocations, displayed on the Tool Log view."""

from datetime import datetime, timezone


class ToolInvocationLog:
    """Append-only invocation log kept for the lifetime of the server process."""

    def __init__(self):
        self._entries = []

    def record(self, tool_name: str, args: dict, result):
        """Appends one invocation.

        Args:
            tool_name: Name of the invoked tool.
            args: Arguments the model passed to the tool.
            result: Tool output (may be None when the run failed mid-call).
        """
        self._entries.append({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool": tool_name,
            "args": args,
            "result": result,
        })

    def entries(self) -> list:
        return list(self._entries)

    def clear(self):
        self._entries.clear()
