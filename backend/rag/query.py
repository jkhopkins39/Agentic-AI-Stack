import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from .document_processor import generate_data_store, CHROMA_PATH

# Ensure environment variables are loaded
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""


def query_rag(query_text):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
    
    Args:
        query_text (str): The text to query the RAG system with.
    
    Returns:
        tuple: (formatted_response, response_text)
    """
    # Check for OpenAI API key before initializing
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        error_msg = (
            "Error: OPENAI_API_KEY not found in environment variables. "
            "Please ensure your .env file contains OPENAI_API_KEY=your_key_here"
        )
        print(error_msg)
        return (
            "I apologize, but I'm currently unable to access the policy information system. "
            "Please contact support@agenticaistack.com for assistance.",
            None
        )
    
    # YOU MUST - Use same embedding function as before
    embedding_function = OpenAIEmbeddings()

    # Check if database exists, if not, generate it
    if not os.path.exists(CHROMA_PATH):
        generate_data_store()

    # Prepare the database with error handling
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    except Exception as e:
        # Silently regenerate database on error
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        generate_data_store()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
  
    # This searches the chroma vector database for documents most similar to query_text. Limits it to top 5 results
    results = db.similarity_search_with_relevance_scores(query_text, k=5)

    # Enhanced handling for no results or low relevance scores
    if len(results) == 0:
        print(f"⚠️ No results found in RAG database for query: {query_text}")
        return (
            "I couldn't find relevant policy information. Please contact customer support for assistance.",
            None
        )
    
    # Filter results by relevance score (lower threshold to include more context)
    # Lower the threshold to 0.5 to get more context
    filtered_results = [(doc, score) for doc, score in results if score >= 0.5]
    
    if len(filtered_results) == 0:
        print(f"⚠️ All results below relevance threshold. Using all results anyway.")
        filtered_results = results
    
    # Combine context from matching documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
    
    print(f"✓ Found {len(filtered_results)} relevant chunks for policy query")
    print(f"Context preview: {context_text[:200]}...")
    
    fallback_response = None
 
    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
  
    # Initialize OpenAI chat model (will use OPENAI_API_KEY from environment)
    try:
        model = ChatOpenAI()
    except Exception as e:
        error_msg = f"Error initializing OpenAI model: {e}"
        print(error_msg)
        return (
            "I apologize, but I'm currently unable to process your policy question. "
            "Please contact support@agenticaistack.com for assistance.",
            None
        )

    # Generate response text based on the prompt (use invoke instead of deprecated predict)
    messages = [{"role": "user", "content": prompt}]
    response = model.invoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)
 
    # Prepend fallback message if relevance was low
    if fallback_response:
        response_text = fallback_response + response_text + "\n\nIf this doesn't fully answer your question, please contact support@agenticaistack.com for more specific help."
 
    # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in filtered_results]
 
    # Format and return response including generated text and sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text

