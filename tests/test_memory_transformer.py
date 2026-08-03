import pytest
from chatboton.agent import build_memory_note, extract_memory_metadata, parse_json_from_llm
import json

def test_parse_json_from_llm_clean():
    text = '{"include": true, "memory": "User likes Python."}'
    result = parse_json_from_llm(text)
    assert result["include"] is True
    assert result["memory"] == "User likes Python."

def test_parse_json_from_llm_markdown():
    text = '```json\n{"include": false, "memory": ""}\n```'
    result = parse_json_from_llm(text)
    assert result["include"] is False

def test_parse_json_from_llm_with_text():
    text = 'Here is the JSON:\n{"include": true, "memory": "test"}\nHope this helps.'
    result = parse_json_from_llm(text)
    assert result["include"] is True
    assert result["memory"] == "test"

def test_parse_json_from_llm_invalid():
    text = 'Not a json at all'
    with pytest.raises(json.JSONDecodeError):
        parse_json_from_llm(text)

SAMPLE = {
    "include": True,
    "object": "User",
    "subject": "React project work",
    "sentiment": "Positive",
    "topics": ["web development", "frontend"],
    "technologies": ["React"],
    "tags": ["project", "work"],
    "memory": "User is working on a React project and enjoys frontend development."
}

def test_extract_memory_metadata():
    metadata = extract_memory_metadata(SAMPLE)
    assert metadata["object"] == "User"
    assert metadata["subject"] == "React project work"
    assert metadata["sentiment"] == "positive"
    assert metadata["topics"] == ["web development", "frontend"]
    assert metadata["technologies"] == ["React"]
    assert metadata["tags"] == ["project", "work"]

def test_extract_memory_metadata_handles_strings_and_missing():
    metadata = extract_memory_metadata({"topics": "python, ai", "technologies": None})
    assert metadata["topics"] == ["python", "ai"]
    assert metadata["technologies"] == []
    assert metadata["object"] == ""
    assert metadata["tags"] == []

def test_build_memory_note():
    note = build_memory_note(SAMPLE)
    assert "Memory about User — React project work (positive)." in note
    assert "User is working on a React project and enjoys frontend development." in note
    assert "Topics: web development, frontend." in note
    assert "Technologies: React." in note
    assert "Tags: project, work." in note

def test_build_memory_note_minimal():
    note = build_memory_note({"memory": "User likes Python."})
    assert note == "User likes Python."

def test_build_memory_note_empty():
    assert build_memory_note({}) == ""
