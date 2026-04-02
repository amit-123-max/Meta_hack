"""HF Spaces entry-point — delegates to app.main FastAPI application."""
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app  # noqa: F401 — uvicorn imports this module

# Run with: uvicorn app:app --host 0.0.0.0 --port 7860
