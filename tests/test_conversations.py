import sys
from pathlib import Path

# Add backend directory to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from unittest.mock import patch, MagicMock

# Import from the correct modules
from database import create_conversation


@patch("database.connection.psycopg2.connect")
def test_create_conversation_success(mock_connect):
    """Test successful conversation creation"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.execute.return_value = None
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    conv_id = create_conversation("session123", user_id="user1", user_email="user@example.com")

    assert conv_id is not None
    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()


@patch("database.connection.psycopg2.connect", side_effect=Exception("DB error"))
def test_create_conversation_failure(mock_connect):
    """Test conversation creation failure handling"""
    result = create_conversation("bad_session")
    assert result is None
