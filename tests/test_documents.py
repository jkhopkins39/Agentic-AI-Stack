import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock
import main

@patch("main.PyPDFDirectoryLoader")
def test_load_documents_calls_loader(mock_loader):
    mock_instance = MagicMock()
    mock_instance.load.return_value = ["doc1", "doc2"]
    mock_loader.return_value = mock_instance

    docs = main.load_documents()

    mock_loader.assert_called_once()
    mock_instance.load.assert_called_once()
    assert len(docs) == 2
    assert docs == ["doc1", "doc2"]

@patch("main.RecursiveCharacterTextSplitter")
def test_split_text_returns_chunks(mock_splitter):
    mock_instance = MagicMock()
    mock_instance.split_documents.return_value = ["chunk1", "chunk2"]
    mock_splitter.return_value = mock_instance

    result = main.split_text(["fake_doc"])
    assert len(result) == 2
    mock_splitter.assert_called_once()

@patch("main.Chroma.from_documents")
def test_save_to_chroma_creates_new_db(mock_chroma, tmp_path):
    test_dir = tmp_path / "chroma"
    main.CHROMA_PATH = str(test_dir)
    docs = ["chunk1", "chunk2"]

    main.save_to_chroma(docs)

    mock_chroma.assert_called_once()
