import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unittest.mock import patch, MagicMock
import main

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

@patch("main.smtplib.SMTP")
def test_send_information_change_email_success(mock_smtp):
    smtp_instance = MagicMock()
    mock_smtp.return_value = smtp_instance

    result = main.send_information_change_email(["email address"], "test@example.com") # update email

    assert result is True
    smtp_instance.send_message.assert_called_once()

@patch("main.smtplib.SMTP", side_effect=Exception("SMTP error"))
def test_send_order_receipt_email_failure(mock_smtp):
    result = main.send_order_receipt_email(make_order_data())
    assert result is False
