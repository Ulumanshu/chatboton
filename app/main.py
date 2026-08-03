"""FastAPI app for the Chatboton playground.

Routes only translate HTTP to agent calls (thin-endpoint style borrowed from
local_tests_zone). Conversation history lives in the browser; the server keeps
one in-memory ToolInvocationLog for the Tool Log view.
"""

from functools import lru_cache

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.config import get_settings
from chatboton.agent import SYSTEM_PROMPT, create_chatboton_agent
from chatboton.context_state import build_context_text, set_current_context
from chatboton.tool_log import ToolInvocationLog
from chatboton.memory_pipeline import memory_pipeline_loop
from chatboton.connectors.qdrant import QdrantConnector
from chatboton.tools.commit_short_term_memory import commit_short_term_memory
import asyncio

templates = Jinja2Templates(directory="app/templates")
tool_log = ToolInvocationLog()
long_term_memory = QdrantConnector(collection="long_term")


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
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: start the memory pipeline loop
        task = asyncio.create_task(memory_pipeline_loop())
        yield
        # Shutdown: cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    app = FastAPI(title="Chatboton", lifespan=lifespan)

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
        # Expose the real conversation context to in-agent tools (reality_check).
        set_current_context(build_context_text(SYSTEM_PROMPT, input_messages))
        
        # Automatic background memory commitment
        user_request = payload.messages[-1].content
        try:
            # We call the tool function directly. 
            # It's decorated with @tool but can be called as a regular function.
            # We use .invoke or just call it if it was a plain function, 
            # but since it's a langchain tool, calling it directly works or tool.run()
            memory_result = commit_short_term_memory.run(user_request)
            auto_activity = [{
                "tool": "commit_short_term_memory [AUTOMATIC]", 
                "args": {"user_request": user_request}, 
                "result": memory_result
            }]
        except Exception as e:
            auto_activity = [{
                "tool": "commit_short_term_memory [AUTOMATIC]", 
                "args": {"user_request": user_request}, 
                "result": f"Error: {str(e)}"
            }]

        for entry in auto_activity:
            tool_log.record(entry["tool"], entry["args"], entry["result"])

        try:
            result = agent.invoke({"messages": input_messages})
        except Exception as exc:
            return JSONResponse(status_code=502, content={"error": str(exc)})

        activity = extract_tool_activity(result["messages"][len(input_messages):])
        for entry in activity:
            tool_log.record(entry["tool"], entry["args"], entry["result"])

        return {
            "reply": result["messages"][-1].content, 
            "tool_activity": auto_activity + activity
        }

    @app.get("/api/tool_log")
    def get_tool_log():
        return {"entries": tool_log.entries()}

    @app.post("/api/tool_log/clear")
    def clear_tool_log():
        tool_log.clear()
        return {"ok": True}
    
    @app.get("/api/memories")
    def get_memories():
        try:
            if not long_term_memory.client.collection_exists("long_term"):
                return {"entries": []}
            response = long_term_memory.client.scroll(
                collection_name="long_term",
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            points, _ = response
            entries = []
            for p in points:
                entries.append({
                    "id": p.id,
                    "memory": p.payload.get("memory"),
                    "original_request": p.payload.get("original_request"),
                    "metadata": {
                        "object": p.payload.get("object", ""),
                        "subject": p.payload.get("subject", ""),
                        "sentiment": p.payload.get("sentiment", ""),
                        "topics": p.payload.get("topics", []),
                        "technologies": p.payload.get("technologies", []),
                        "tags": p.payload.get("tags", []),
                    }
                })
            return {"entries": entries}
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/api/memories/clear")
    def clear_memories():
        try:
            if long_term_memory.client.collection_exists("long_term"):
                long_term_memory.client.delete_collection("long_term")
            return {"ok": True}
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})
    
    return app


app = create_app()
