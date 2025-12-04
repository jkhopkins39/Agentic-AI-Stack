import sys
from pathlib import Path

# Add backend directory to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from unittest.mock import patch, MagicMock

# Import functions from the correct modules
from agents import classify_message, router


@patch("agents.handlers.llm")
def test_classify_message_order(mock_llm):
    """Test that classify_message correctly identifies an order query"""
    # Create a mock structured output
    mock_structured_llm = MagicMock()
    mock_result = MagicMock()
    mock_result.message_type = "Order"
    mock_structured_llm.invoke.return_value = mock_result
    mock_llm.with_structured_output.return_value = mock_structured_llm

    # FIX: The message needs to be a proper message object with 'content' attribute
    # Create a mock message object
    mock_message = MagicMock()
    mock_message.content = "Where is my order?"
    
    state = {"messages": [mock_message]}
    result = classify_message(state)

    assert "message_type" in result
    assert result["message_type"] == "Order"


def test_router_returns_expected_next():
    """Test that router correctly routes to policy agent"""
    state = {"message_type": "Policy"}
    result = router(state)
    assert result == {"next": "policy"}
