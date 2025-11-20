import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unittest.mock import patch, MagicMock
import main

@patch("main.psycopg2.connect")
def test_create_conversation_success(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.execute.return_value = None
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    conv_id = main.create_conversation("session123", user_id="user1", user_email="user@example.com") # swap to real session info

    assert conv_id is not None
    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()

@patch("main.psycopg2.connect", side_effect=Exception("DB error"))
def test_create_conversation_failure(mock_connect):
    result = main.create_conversation("bad_session")
    assert result is None
