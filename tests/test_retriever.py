"""
Tests for the dense retriever (src/retriever.py).

Unit tests mock the embedder and store — no model needed.
Integration tests use the session-scoped embedder from conftest.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.chunker import Chunk
from src.retriever import Retriever
from src.vector_store import VectorStore

DIMENSION = 384


def make_chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(text=text, index=index, start_char=0, end_char=len(text),
                 metadata={"source": "test"})


def make_chunks(texts: list[str]) -> list[Chunk]:
    return [make_chunk(t, i) for i, t in enumerate(texts)]


def rand_embs(n: int, dim: int = DIMENSION) -> np.ndarray:
    rng = np.random.default_rng(0)
    vecs = rng.random((n, dim), dtype=np.float32)
    return (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)


def build_store(texts: list[str], embedder) -> VectorStore:
    chunks = make_chunks(texts)
    embeddings = embedder.embed(texts)
    store = VectorStore(embedder.dimension)
    store.add(chunks, embeddings)
    return store


# ---------------------------------------------------------------------------
# Unit tests — mocked embedder + store
# ---------------------------------------------------------------------------

class TestRetrieverUnit:
    def _make(self, results):
        mock_embedder = MagicMock()
        mock_embedder.embed_one.return_value = rand_embs(1)[0]
        mock_store = MagicMock()
        mock_store.search.return_value = results
        return Retriever(mock_embedder, mock_store)

    def test_returns_list(self):
        r = self._make([(make_chunk("hello"), 0.9)])
        assert isinstance(r.retrieve("test query"), list)

    def test_empty_query_raises(self):
        r = self._make([])
        with pytest.raises(ValueError):
            r.retrieve("   ")

    def test_calls_embed_one_with_query(self):
        chunk = make_chunk("hello")
        r = self._make([(chunk, 0.8)])
        r.retrieve("my question")
        r.embedder.embed_one.assert_called_once_with("my question")

    def test_calls_store_search_with_k(self):
        r = self._make([(make_chunk("hello"), 0.8)])
        r.retrieve("question", k=7)
        call = r.store.search.call_args
        k_value = call.kwargs.get("k") or (call.args[1] if len(call.args) > 1 else None)
        assert k_value == 7

    def test_min_score_filters_low_results(self):
        results = [
            (make_chunk("good"), 0.8),
            (make_chunk("weak"), 0.2),
            (make_chunk("ok"),   0.5),
        ]
        r = self._make(results)
        filtered = r.retrieve("question", min_score=0.4)
        assert all(s >= 0.4 for _, s in filtered)
        assert len(filtered) == 2

    def test_min_score_zero_returns_all(self):
        results = [(make_chunk("a"), 0.1), (make_chunk("b"), 0.9)]
        r = self._make(results)
        assert len(r.retrieve("question", min_score=0.0)) == 2

    def test_results_are_chunk_score_tuples(self):
        chunk = make_chunk("hello world")
        r = self._make([(chunk, 0.75)])
        c, s = r.retrieve("hello")[0]
        assert isinstance(c, Chunk)
        assert isinstance(s, float)


# ---------------------------------------------------------------------------
# Integration tests — real embedder
# ---------------------------------------------------------------------------

class TestRetrieverIntegration:
    def test_retrieves_relevant_chunk(self, embedder):
        texts = [
            "The Eiffel Tower is in Paris, France.",
            "Python is a high-level programming language.",
            "Photosynthesis converts sunlight into energy.",
        ]
        store = build_store(texts, embedder)
        retriever = Retriever(embedder, store)
        results = retriever.retrieve("Where is the Eiffel Tower?", k=1)
        assert len(results) == 1
        assert "Paris" in results[0][0].text or "Eiffel" in results[0][0].text

    def test_k_limits_results(self, embedder):
        texts = [f"Sentence number {i}." for i in range(10)]
        store = build_store(texts, embedder)
        retriever = Retriever(embedder, store)
        assert len(retriever.retrieve("sentence", k=3)) == 3

    def test_results_sorted_descending(self, embedder):
        texts = [
            "Dogs are loyal and friendly pets.",
            "Cats are independent animals.",
            "Quantum physics studies subatomic particles.",
        ]
        store = build_store(texts, embedder)
        retriever = Retriever(embedder, store)
        scores = [s for _, s in retriever.retrieve("What makes a good pet?", k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filters_irrelevant(self, embedder):
        texts = [
            "The mitochondria is the powerhouse of the cell.",
            "Paris is the capital of France.",
        ]
        store = build_store(texts, embedder)
        retriever = Retriever(embedder, store)
        results = retriever.retrieve("cell biology", k=2, min_score=0.99)
        assert all(s >= 0.99 for _, s in results)


# ---------------------------------------------------------------------------
# Full pipeline integration — retriever → prompt → generator
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    def test_pipeline_produces_grounded_prompt(self, embedder):
        from src.prompt import build_prompt, get_system_prompt

        texts = [
            "The speed of light is approximately 299,792 kilometres per second.",
            "Albert Einstein developed the theory of relativity.",
            "Water is composed of hydrogen and oxygen atoms.",
        ]
        store = build_store(texts, embedder)
        retriever = Retriever(embedder, store)

        query = "How fast does light travel?"
        results = retriever.retrieve(query, k=2)
        prompt = build_prompt(query, results)

        assert "299,792" in prompt or "speed of light" in prompt.lower()
        assert query in prompt
        assert "[Context]" in prompt
        assert "[Question]" in prompt
