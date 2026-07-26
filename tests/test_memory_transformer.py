import pytest
from chatboton.agent import parse_json_from_llm
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
