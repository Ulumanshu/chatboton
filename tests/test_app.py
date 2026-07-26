"""App endpoint tests with a fake agent — no LLM, no databases."""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import main


class FakeAgent:
    """Echoes a canned tool-calling exchange, like the real LangChain agent."""

    def __init__(self, fail=False):
        self.fail = fail

    def invoke(self, payload):
        if self.fail:
            raise RuntimeError("model exploded")
        input_messages = [HumanMessage(content=m["content"]) for m in payload["messages"]]
        return {
            "messages": input_messages + [
                AIMessage(content="", tool_calls=[
                    {"name": "query_postgres", "args": {"sql": "SELECT 1"}, "id": "call-1"},
                ]),
                ToolMessage(content='[{"?column?": 1}]', tool_call_id="call-1"),
                AIMessage(content="The answer is 1."),
            ],
        }


@pytest.fixture
def client(monkeypatch):
    main.tool_log.clear()
    monkeypatch.setattr(main, "get_agent", lambda: FakeAgent())
    return TestClient(main.app)


def test_index_serves_both_views(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Chatboton" in response.text.lower() or "CHATBOTON" in response.text
    assert 'id="view-chat"' in response.text
    assert 'id="view-log"' in response.text
    assert 'id="reset-btn"' in response.text


def test_chat_returns_reply_and_tool_activity(client):
    response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "The answer is 1."
    assert data["tool_activity"] == [
        {
            "tool": "commit_short_term_memory [AUTOMATIC]",
            "args": {"user_request": "hi"},
            "result": "Success",
        },
        {
            "tool": "query_postgres",
            "args": {"sql": "SELECT 1"},
            "result": '[{"?column?": 1}]',
        }
    ]


def test_chat_records_invocations_in_tool_log(client):
    client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    client.post("/api/chat", json={"messages": [{"role": "user", "content": "again"}]})
    entries = client.get("/api/tool_log").json()["entries"]
    assert len(entries) == 4
    assert entries[0]["tool"] == "commit_short_term_memory [AUTOMATIC]"
    assert entries[1]["tool"] == "query_postgres"
    assert entries[1]["result"] == '[{"?column?": 1}]'


def test_tool_log_clear(client):
    client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert client.post("/api/tool_log/clear").json() == {"ok": True}
    assert client.get("/api/tool_log").json()["entries"] == []


def test_chat_agent_failure_returns_502(client, monkeypatch):
    monkeypatch.setattr(main, "get_agent", lambda: FakeAgent(fail=True))
    response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 502
    assert "model exploded" in response.json()["error"]


def test_chat_rejects_empty_history(client):
    assert client.post("/api/chat", json={"messages": []}).status_code == 422
