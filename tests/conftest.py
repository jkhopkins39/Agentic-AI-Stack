"""
Pytest configuration file
This file is automatically loaded by pytest and sets up the Python path
so tests can import from the backend module
"""
import sys
from pathlib import Path

# Add backend directory to Python path
# This allows: from agents import X, from database import Y, etc.
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

print(f"✓ Backend path added to sys.path: {backend_path}")
