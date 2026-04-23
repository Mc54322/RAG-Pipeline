"""
Tests for the FAISS-backed vector store (src/vector_store.py).

Unit tests use random normalised vectors — no model needed.
Integration tests use the session-scoped embedder from conftest.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.chunker import Chunk, chunk_text
from src.vector_store import VectorStore

DIMENSION = 384


def make_chunks(texts: list[str], source: str = "test") -> list[Chunk]:
    return [
        Chunk(text=t, index=i, start_char=0, end_char=len(t), metadata={"source": source})
        for i, t in enumerate(texts)
    ]


def rand_embs(n: int, dim: int = DIMENSION) -> np.ndarray:
    """Return L2-normalised random float32 embeddings."""
    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Unit tests — no model needed
# ---------------------------------------------------------------------------

class TestVectorStoreUnit:
    def test_empty_store_len_zero(self):
        assert len(VectorStore(DIMENSION)) == 0

    def test_add_updates_len(self):
        store = VectorStore(DIMENSION)
        store.add(make_chunks(["a", "b", "c"]), rand_embs(3))
        assert len(store) == 3

    def test_add_accumulates(self):
        store = VectorStore(DIMENSION)
        store.add(make_chunks(["a", "b"]), rand_embs(2))
        store.add(make_chunks(["c"]), rand_embs(1))
        assert len(store) == 3

    def test_search_returns_k_results(self):
        store = VectorStore(DIMENSION)
        store.add(make_chunks(["a", "b", "c", "d", "e"]), rand_embs(5))
        assert len(store.search(rand_embs(1)[0], k=3)) == 3

    def test_search_result_type(self):
        store = VectorStore(DIMENSION)
        store.add(make_chunks(["hello"]), rand_embs(1))
        chunk, score = store.search(rand_embs(1)[0], k=1)[0]
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)

    def test_search_k_capped_at_store_size(self):
        store = VectorStore(DIMENSION)
        store.add(make_chunks(["only one"]), rand_embs(1))
        assert len(store.search(rand_embs(1)[0], k=10)) == 1

    def test_search_empty_store_raises(self):
        store = VectorStore(DIMENSION)
        with pytest.raises(ValueError):
            store.search(rand_embs(1)[0])

    def test_add_mismatched_lengths_raises(self):
        store = VectorStore(DIMENSION)
        with pytest.raises(ValueError):
            store.add(make_chunks(["a", "b"]), rand_embs(3))

    def test_add_wrong_dimension_raises(self):
        store = VectorStore(DIMENSION)
        with pytest.raises(ValueError):
            store.add(make_chunks(["a", "b"]), rand_embs(2, dim=128))

    def test_search_accepts_2d_query(self):
        store = VectorStore(DIMENSION)
        store.add(make_chunks(["hello"]), rand_embs(1))
        assert len(store.search(rand_embs(1), k=1)) == 1  # shape (1, 384)

    def test_scores_in_valid_range(self):
        """Cosine similarities between normalised vectors are in [-1, 1]."""
        store = VectorStore(DIMENSION)
        store.add(make_chunks(["a", "b", "c"]), rand_embs(3))
        for _, score in store.search(rand_embs(1)[0], k=3):
            assert -1.0 <= score <= 1.0

    def test_results_sorted_descending(self):
        store = VectorStore(DIMENSION)
        store.add(make_chunks(["a", "b", "c", "d"]), rand_embs(4))
        scores = [s for _, s in store.search(rand_embs(1)[0], k=4)]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestVectorStorePersistence:
    def test_save_creates_files(self, tmp_path):
        store = VectorStore(DIMENSION)
        store.add(make_chunks(["hello"]), rand_embs(1))
        store.save(tmp_path / "store")
        assert (tmp_path / "store" / "index.faiss").exists()
        assert (tmp_path / "store" / "chunks.pkl").exists()

    def test_load_restores_length(self, tmp_path):
        store = VectorStore(DIMENSION)
        store.add(make_chunks(["a", "b", "c"]), rand_embs(3))
        store.save(tmp_path / "store")
        assert len(VectorStore.load(tmp_path / "store")) == 3

    def test_load_restores_chunks(self, tmp_path):
        store = VectorStore(DIMENSION)
        store.add(make_chunks(["hello world"]), rand_embs(1))
        store.save(tmp_path / "store")
        loaded = VectorStore.load(tmp_path / "store")
        assert loaded.chunks[0].text == "hello world"

    def test_load_restores_search(self, tmp_path):
        embs = rand_embs(3)
        store = VectorStore(DIMENSION)
        store.add(make_chunks(["a", "b", "c"]), embs)
        store.save(tmp_path / "store")
        loaded = VectorStore.load(tmp_path / "store")
        assert loaded.search(embs[0], k=1)[0][0].text == "a"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VectorStore.load(tmp_path / "nonexistent")

    def test_save_creates_nested_directory(self, tmp_path):
        store = VectorStore(DIMENSION)
        store.add(make_chunks(["x"]), rand_embs(1))
        store.save(tmp_path / "nested" / "store")
        assert (tmp_path / "nested" / "store" / "index.faiss").exists()


# ---------------------------------------------------------------------------
# Integration — real embedder + vector store end-to-end
# ---------------------------------------------------------------------------

class TestVectorStoreIntegration:
    def test_query_retrieves_relevant_chunk(self, embedder):
        texts = [
            "Paris is the capital city of France.",
            "The Eiffel Tower is located in Paris.",
            "Python is a popular programming language.",
            "Neural networks are inspired by the brain.",
        ]
        chunks = make_chunks(texts)
        embeddings = embedder.embed(texts)
        store = VectorStore(embedder.dimension)
        store.add(chunks, embeddings)

        query = embedder.embed_one("What is the capital of France?")
        top = store.search(query, k=2)[0][0]
        assert "Paris" in top.text or "France" in top.text

    def test_end_to_end_with_chunker(self, embedder):
        text = (
            "Machine learning is a subset of artificial intelligence. "
            "It allows computers to learn from data without being explicitly programmed. "
            "Deep learning uses neural networks with many layers. "
            "Natural language processing enables machines to understand text."
        )
        chunks = chunk_text(text, chunk_size=100, overlap=20, source="sample")
        embeddings = embedder.embed([c.text for c in chunks])
        store = VectorStore(embedder.dimension)
        store.add(chunks, embeddings)

        results = store.search(embedder.embed_one("neural networks and deep learning"), k=2)
        combined = " ".join(r[0].text for r in results)
        assert "neural" in combined.lower() or "deep" in combined.lower()
