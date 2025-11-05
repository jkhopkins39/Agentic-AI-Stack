"""
Main entry point for the LangGraph AI Chatbot System.

This modular chatbot system provides:
- Order management and receipt generation
- Policy queries using RAG (Retrieval-Augmented Generation)
- User information updates
- Multi-channel notifications (Email & SMS)
- Conversation tracking and context memory
"""
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from graph import run_chatbot

# Suppress LangChain deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# Load environment variables from .env file in project root
# Get the project root (parent of backend directory)
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Verify required API keys are set
required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
missing_keys = [key for key in required_keys if not os.getenv(key)]

if missing_keys:
    print("Error: Missing required environment variables:")
    for key in missing_keys:
        print(f"  - {key}")
    print(f"\nPlease ensure your .env file exists at: {env_path}")
    print("And contains the required API keys.")
    exit(1)

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
