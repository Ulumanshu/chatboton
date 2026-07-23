"""FastAPI app for the Chatboton playground.

Routes only translate HTTP to agent calls (thin-endpoint style borrowed from
local_tests_zone). Conversation history lives in the browser; the server keeps
one in-memory ToolInvocationLog for the Tool Log view.
"""

from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.config import get_settings
from chatboton.agent import create_chatboton_agent
from chatboton.tool_log import ToolInvocationLog

templates = Jinja2Templates(directory="app/templates")
tool_log = ToolInvocationLog()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


@lru_cache
def get_agent():
    get_settings()  # ensures .env is loaded before the provider reads os.environ
    return create_chatboton_agent()


def extract_tool_activity(new_messages) -> list:
    """Pairs tool calls with their results from the agent's new messages."""
    activity = []
    by_call_id = {}
    for message in new_messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            entry = {"tool": tool_call["name"], "args": tool_call["args"], "result": None}
            by_call_id[tool_call.get("id")] = entry
            activity.append(entry)
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id in by_call_id:
            by_call_id[tool_call_id]["result"] = message.content
    return activity


def create_app() -> FastAPI:
    app = FastAPI(title="Chatboton")

    @app.get("/")
    def index(request: Request):
        settings = get_settings()
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "provider": settings.chatboton_provider,
                "model_name": settings.ollama_model,
            },
        )

    @app.post("/api/chat")
    def chat(payload: ChatRequest):
        agent = get_agent()
        input_messages = [
            {"role": message.role, "content": message.content}
            for message in payload.messages
        ]
        try:
            result = agent.invoke({"messages": input_messages})
        except Exception as exc:
            return JSONResponse(status_code=502, content={"error": str(exc)})

        activity = extract_tool_activity(result["messages"][len(input_messages):])
        for entry in activity:
            tool_log.record(entry["tool"], entry["args"], entry["result"])

        return {"reply": result["messages"][-1].content, "tool_activity": activity}

    @app.get("/api/tool_log")
    def get_tool_log():
        return {"entries": tool_log.entries()}

    @app.post("/api/tool_log/clear")
    def clear_tool_log():
        tool_log.clear()
        return {"ok": True}

    return app


app = create_app()
