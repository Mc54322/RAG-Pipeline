"""
Shared pytest fixtures for the RAG pipeline test suite.

The session-scoped fixtures here are loaded once per test session and
shared across all test modules, avoiding redundant model loading.

  embedder — all-MiniLM-L6-v2 (~80 MB, ~2 s startup)
  reranker — cross-encoder/ms-marco-MiniLM-L-6-v2 (~90 MB, ~3 s startup)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture(scope="session")
def embedder():
    """Load the sentence-transformer model once for the entire session."""
    from src.embedder import Embedder
    return Embedder()


@pytest.fixture(scope="session")
def reranker():
    """Load the cross-encoder model once for the entire session."""
    from src.reranker import Reranker
    return Reranker()
