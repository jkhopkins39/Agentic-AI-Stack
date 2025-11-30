import sys
from pathlib import Path

# Add backend directory to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

import pytest
from unittest.mock import patch, MagicMock

# Import from the correct modules
from rag import load_documents, split_text, save_to_chroma
import rag.document_processor as doc_processor


@patch("rag.document_processor.PyPDFDirectoryLoader")
def test_load_documents_calls_loader(mock_loader):
    """Test that load_documents properly calls the PDF loader"""
    mock_instance = MagicMock()
    mock_instance.load.return_value = ["doc1", "doc2"]
    mock_loader.return_value = mock_instance

    docs = load_documents()

    mock_loader.assert_called_once()
    mock_instance.load.assert_called_once()
    assert len(docs) == 2
    assert docs == ["doc1", "doc2"]


@patch("rag.document_processor.RecursiveCharacterTextSplitter")
def test_split_text_returns_chunks(mock_splitter):
    """Test that split_text properly chunks documents"""
    mock_instance = MagicMock()
    mock_instance.split_documents.return_value = ["chunk1", "chunk2"]
    mock_splitter.return_value = mock_instance

    result = split_text(["fake_doc"])
    
    assert len(result) == 2
    mock_splitter.assert_called_once()


@patch("rag.document_processor.Chroma.from_documents")
def test_save_to_chroma_creates_new_db(mock_chroma, tmp_path):
    """Test that save_to_chroma creates a new Chroma database"""
    test_dir = tmp_path / "chroma"
    doc_processor.CHROMA_PATH = str(test_dir)
    docs = ["chunk1", "chunk2"]

    save_to_chroma(docs)

    mock_chroma.assert_called_once()
