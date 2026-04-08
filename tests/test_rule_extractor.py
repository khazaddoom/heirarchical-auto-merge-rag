import pytest
from unittest.mock import patch, MagicMock
import json

from src.rule_extractor import extract_rules_for_category
from src.rule_schema import ContractRule

def test_extract_rules_for_category_success():
    # Mock pipeline instance
    mock_pipe = MagicMock()
    
    # Simulate a perfect JSON response from OpenAI generating exactly our desired structure!
    fake_llm_json = {
        "rules": [
            {
                "category": "Travel Policies",
                "rule_statement": "Travel rules are handled according to an external schedule.",
                "trigger_condition": "Employee requires flight travel.",
                "action_or_limit": "Reimburse based on table.",
                "requires_approval": True,
                "requires_external_lookup": True,
                "external_reference_source": "Appendix D"
            }
        ]
    }
    
    mock_pipe.run.return_value = {
        "llm": {
            "replies": [json.dumps(fake_llm_json)]
        }
    }
    
    rules = extract_rules_for_category(mock_pipe, "Travel Policies")
    
    assert len(rules) == 1
    assert rules[0]["requires_external_lookup"] == True
    assert rules[0]["external_reference_source"] == "Appendix D"

def test_extract_rules_for_category_validation_error():
    mock_pipe = MagicMock()
    
    # Missing required 'rule_statement' logic, triggering Pydantic validation error!
    bad_json = {
        "rules": [
            {
                "category": "Travel Policies"
            }
        ]
    }
    
    mock_pipe.run.return_value = {
        "llm": {
            "replies": [json.dumps(bad_json)]
        }
    }
    
    # Should safely return empty list instead of crashing!
    rules = extract_rules_for_category(mock_pipe, "Travel Policies")
    assert rules == []
