import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from unittest.mock import patch, MagicMock
import main

@patch("main.psycopg2.connect")
def test_lookup_user_by_email_success(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = { # change to real email account
        "id": 1, "email": "user@example.com", "first_name": "John", "last_name": "Doe"
    }
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    result = main.lookup_user_by_email("user@example.com") # check here

    assert result["email"] == "user@example.com" # check here
    mock_cursor.execute.assert_called_once()
    mock_conn.close.assert_called_once()

@patch("main.psycopg2.connect", side_effect=Exception("DB error"))
def test_get_database_connection_failure(mock_connect):
    result = main.get_database_connection()
    assert result is None
