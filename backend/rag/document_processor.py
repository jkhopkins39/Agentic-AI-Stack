import os
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# Ensure environment variables are loaded
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Data path reads in txt file for policy RAG
DATA_PATH = os.path.join(os.getcwd())
CHROMA_PATH = "chroma"


def load_documents():
    """Load PDF documents from DATA_PATH"""
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    # Debug print removed for cleaner output
    return documents


def split_text(documents: list[Document]):
    """Split documents into chunks for processing"""
    # Initialize text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Size of each chunk in characters
        length_function=len,  # Function to compute the length of the text
        chunk_overlap=100,  # Overlap between consec chunks
        add_start_index=True,  # Flag to add start index to each chunk
    )

    # Make our list of chunks of text, could handle splitting of multiple documents
    chunks = text_splitter.split_documents(documents)
    # Debug prints removed for cleaner output

    return chunks


def save_to_chroma(chunks: list[Document]):
    """
    Save the given list of Document objects to a Chroma database.
    
    Args:
        chunks (list[Document]): List of Document objects representing text chunks to save.
    """
    # Check for OpenAI API key before initializing
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        error_msg = (
            "Error: OPENAI_API_KEY not found in environment variables. "
            "Please ensure your .env file contains OPENAI_API_KEY=your_key_here"
        )
        print(error_msg)
        raise ValueError(error_msg)
    
    # Clear out the existing database directory if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        time.sleep(1)  # Gives filesystem time to clean up
  
    # Ensure the directory is completely gone
    while os.path.exists(CHROMA_PATH):
        time.sleep(0.5)
  
    try:
        # Create a new Chroma database from the documents using OpenAI embeddings
        db = Chroma.from_documents(
            chunks,
            OpenAIEmbeddings(),
            persist_directory=CHROMA_PATH
        )
        # Debug print removed for cleaner output
    
    except Exception as e:
        print(f"Error creating database: {e}")
        # If there's an error, clean up any partial database
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        raise


def load_policy_text():
    """Load policy as a document from txt file"""
    policy_path = "policy.txt"
    if os.path.exists(policy_path):
        with open(policy_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [Document(page_content=content, metadata={"source": policy_path})]
    return []


def generate_data_store():
    """Generate the complete data store from all documents"""
    documents = load_documents()  # Load documents from a source
    policy_docs = load_policy_text()  # Load policy text
    all_documents = documents + policy_docs  # Combine all documents
    chunks = split_text(all_documents)  # Split documents into chunks
    save_to_chroma(chunks)  # Save the processed data to a data store

