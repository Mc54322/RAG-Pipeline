"""
Tests for BM25 keyword retrieval (src/bm25_store.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.bm25_store import BM25Store
from src.chunker import Chunk


def make_chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, index=idx, start_char=0, end_char=len(text), metadata={})


def make_chunks(*texts: str) -> list[Chunk]:
    return [make_chunk(t, i) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

class TestBM25Tokenize:
    def test_lowercase(self):
        assert BM25Store.tokenize("Hello World") == ["hello", "world"]

    def test_splits_on_punctuation(self):
        tokens = BM25Store.tokenize("gradient-descent, loss.function")
        assert "gradient" in tokens
        assert "descent" in tokens
        assert "loss" in tokens
        assert "function" in tokens

    def test_drops_single_chars(self):
        tokens = BM25Store.tokenize("a b c python")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "python" in tokens

    def test_empty_string(self):
        assert BM25Store.tokenize("") == []

    def test_only_punctuation(self):
        assert BM25Store.tokenize("!!! ???") == []

    def test_numbers_retained(self):
        tokens = BM25Store.tokenize("1789 revolution")
        assert "1789" in tokens
        assert "revolution" in tokens


# ---------------------------------------------------------------------------
# Construction and search
# ---------------------------------------------------------------------------

class TestBM25StoreSearch:
    def test_len(self):
        store = BM25Store(make_chunks("Python is great.", "Climate change."))
        assert len(store) == 2

    def test_returns_list_of_tuples(self):
        store = BM25Store(make_chunks("Python is popular.", "Climate change is real."))
        results = store.search("Python", k=2)
        assert isinstance(results, list)
        assert all(isinstance(c, Chunk) and isinstance(s, float) for c, s in results)

    def test_relevant_chunk_ranked_first(self):
        chunks = make_chunks(
            "Gradient descent optimises neural networks.",
            "The French Revolution began in 1789.",
            "Python uses NumPy for array operations.",
        )
        store = BM25Store(chunks)
        top, _ = store.search("gradient descent neural", k=3)[0]
        assert "gradient" in top.text.lower()

    def test_k_limits_results(self):
        store = BM25Store(make_chunks("A alpha.", "B beta.", "C gamma.", "D delta."))
        assert len(store.search("alpha beta gamma delta", k=2)) <= 2

    def test_k_larger_than_corpus(self):
        store = BM25Store(make_chunks("Only one chunk here."))
        assert len(store.search("chunk", k=10)) == 1

    def test_empty_query_returns_empty(self):
        store = BM25Store(make_chunks("Python is great."))
        assert store.search("!!! ???", k=5) == []

    def test_empty_store_raises(self):
        with pytest.raises(ValueError):
            BM25Store([])

    def test_irrelevant_query_scores_zero(self):
        store = BM25Store(make_chunks("Python programming language.",
                                     "Machine learning algorithms."))
        for _, score in store.search("zzzzz", k=2):
            assert score == pytest.approx(0.0)

    def test_score_descending_order(self):
        store = BM25Store(make_chunks(
            "Python python python",
            "Python is great",
            "The French Revolution",
        ))
        scores = [s for _, s in store.search("python", k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_chunk_objects_preserved(self):
        """Results contain the same Chunk objects (identity preserved)."""
        chunks = make_chunks("Python is popular.", "DNA double helix.")
        store = BM25Store(chunks)
        result_chunks = [c for c, _ in store.search("python", k=2)]
        assert any(c is chunks[0] for c in result_chunks)
