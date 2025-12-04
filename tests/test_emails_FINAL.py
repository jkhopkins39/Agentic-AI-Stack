import sys
from pathlib import Path

# Add backend directory to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Import from the correct modules
from notifications import send_information_change_email, send_order_receipt_email


# UPDATE WITH FAKER RANDOMIZED DATA
def make_order_data():
    return {
        "order_number": "ORD-001",
        "first_name": "John",
        "last_name": "Doe",
        "email": "john@example.com",
        "created_at": None,
        "status": "shipped",
        "items": [{"product_name": "Widget", "quantity": 1, "unit_price": 10.0, "total_price": 10.0}],
        "total_amount": 10.0,
        "currency": "USD"
    }


@pytest.mark.asyncio
@patch("notifications.mailersend.send_email_via_mailersend")
async def test_send_information_change_email_success(mock_send_email):
    """Test successful information change email sending"""
    # Mock the async function to return True
    mock_send_email.return_value = True

    # FIX: Email functions are async, so we need to await them
    result = await send_information_change_email(["email address"], "test@example.com")

    assert result is True
    mock_send_email.assert_called_once()


@pytest.mark.asyncio
@patch("notifications.mailersend.send_email_via_mailersend", side_effect=Exception("Email error"))
async def test_send_order_receipt_email_failure(mock_send_email):
    """Test order receipt email sending failure handling"""
    # FIX: Email functions are async, so we need to await them
    result = await send_order_receipt_email(make_order_data())
    assert result is False
