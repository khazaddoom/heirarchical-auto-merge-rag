import pytest
from unittest.mock import patch, MagicMock
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from src.category_discovery import CategoryDiscoveryEngine

def test_category_discovery_map_reduce():
    doc_store = InMemoryDocumentStore()
    docs = [
        Document(content="The contractor will bill 1.5x for all hours worked over 40.", meta={"__level": 1}),
        Document(content="Suncor will reimburse flight tickets and per diem meals.", meta={"__level": 1})
    ]
    doc_store.write_documents(docs)

    with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
        engine = CategoryDiscoveryEngine()
        
    # Mock the pipeline runs directly to bypass LLM and preserve Haystack construction
    def mock_map_run(data):
        prompt = data["prompt"]["text"]
        if "hours" in prompt:
            return {"llm": {"replies": ["- Labor & Overtime Rates"]}}
        elif "flight" in prompt:
            return {"llm": {"replies": ["- Travel & Per Diem Policies"]}}
        return {"llm": {"replies": ["NONE"]}}
        
    def mock_reduce_run(data):
        return {"llm": {"replies": ["1. Labor & Overtime Rates\n2. Travel & Per Diem Policies"]}}
        
    engine.map_pipe.run = MagicMock(side_effect=mock_map_run)
    engine.reduce_pipe.run = MagicMock(side_effect=mock_reduce_run)
        
    categories = engine.discover_categories(doc_store)

    assert len(categories) == 2
    assert "Labor & Overtime Rates" in categories[0]
    assert "Travel & Per Diem Policies" in categories[1]
