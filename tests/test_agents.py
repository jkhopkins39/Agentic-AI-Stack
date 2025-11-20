import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unittest.mock import MagicMock, patch
import main

@patch("main.llm.with_structured_output")
def test_classify_message_order(mock_structured):
    mock_classifier = MagicMock()
    mock_classifier.invoke.return_value.message_type = "Order"
    mock_structured.return_value = mock_classifier

    state = {"messages": [{"content": "Where is my order?"}]}
    result = main.classify_message(state)

    assert "message_type" in result
    assert result["message_type"] == "Order"

def test_router_returns_expected_next():
    state = {"message_type": "Policy"}
    result = main.router(state)
    assert result == {"next": "policy"}
