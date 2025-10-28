"""Document loading, splitting, and storage for RAG system."""
import os
import shutil
import time
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma

# Data path reads in txt file for policy RAG
DATA_PATH = os.path.join(os.getcwd())
CHROMA_PATH = "chroma"


def load_documents():
    """Load PDF documents from DATA_PATH"""
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    print(f"Loaded {len(documents)} documents from {DATA_PATH}")
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
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    if chunks:
        document = chunks[0]
        # This is so we can visualize what just happened and what was split and how
        print(document.page_content)
        print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    """
    Save the given list of Document objects to a Chroma database.
    
    Args:
        chunks (list[Document]): List of Document objects representing text chunks to save.
    """
    # Clear out the existing database directory if it exists
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing database at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
        time.sleep(1)  # Gives filesystem time to clean up
  
    # Ensure the directory is completely gone
    while os.path.exists(CHROMA_PATH):
        time.sleep(0.5)
  
    print(f"Creating new database with {len(chunks)} chunks...")
  
    try:
        # Create a new Chroma database from the documents using OpenAI embeddings
        db = Chroma.from_documents(
            chunks,
            OpenAIEmbeddings(),
            persist_directory=CHROMA_PATH
        )
        print(f"Successfully saved {len(chunks)} chunks to {CHROMA_PATH}.")
    
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

