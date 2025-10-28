"""
Main entry point for the LangGraph AI Chatbot System.

This modular chatbot system provides:
- Order management and receipt generation
- Policy queries using RAG (Retrieval-Augmented Generation)
- User information updates
- Multi-channel notifications (Email & SMS)
- Conversation tracking and context memory
"""
from dotenv import load_dotenv
from graph import run_chatbot

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    run_chatbot()

# Uncomment below to generate graph visualization (requires IPython)
'''
from IPython.display import Image, display
from graph import graph

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass
'''
