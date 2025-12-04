import sys
from pathlib import Path

# Add backend directory to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

import pytest
from unittest.mock import patch, MagicMock

# Import from the correct modules
from database import lookup_user_by_email, get_database_connection


@patch("database.connection.psycopg2.connect")
def test_lookup_user_by_email_success(mock_connect):
    """Test successful user lookup by email"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {
        "id": 1, "email": "user@example.com", "first_name": "John", "last_name": "Doe"
    }
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    result = lookup_user_by_email("user@example.com")

    assert result["email"] == "user@example.com"
    mock_cursor.execute.assert_called_once()
    mock_conn.close.assert_called_once()


@patch("database.connection.psycopg2.connect", side_effect=Exception("DB error"))
def test_get_database_connection_failure(mock_connect):
    """Test database connection failure handling"""
    result = get_database_connection()
    assert result is None
