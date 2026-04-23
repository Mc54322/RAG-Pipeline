"""
Tests for the text embedding module (src/embedder.py).

The session-scoped `embedder` fixture is provided by conftest.py and loads
the model once for the entire test session.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

DIMENSION = 384  # all-MiniLM-L6-v2 output size


# ---------------------------------------------------------------------------
# Shape and type contracts
# ---------------------------------------------------------------------------

class TestEmbedderShape:
    def test_embed_returns_2d_array(self, embedder):
        assert embedder.embed(["hello world"]).ndim == 2

    def test_embed_correct_shape(self, embedder):
        result = embedder.embed(["first sentence", "second sentence", "third sentence"])
        assert result.shape == (3, DIMENSION)

    def test_embed_dtype_float32(self, embedder):
        assert embedder.embed(["hello"]).dtype == np.float32

    def test_embed_one_returns_1d(self, embedder):
        result = embedder.embed_one("hello world")
        assert result.ndim == 1
        assert result.shape == (DIMENSION,)

    def test_embed_one_dtype_float32(self, embedder):
        assert embedder.embed_one("hello").dtype == np.float32

    def test_empty_list_raises(self, embedder):
        with pytest.raises(ValueError):
            embedder.embed([])

    def test_dimension_attribute(self, embedder):
        assert embedder.dimension == DIMENSION


# ---------------------------------------------------------------------------
# Semantic correctness
# ---------------------------------------------------------------------------

class TestEmbedderSemantics:
    def test_unit_length_vectors(self, embedder):
        """All returned vectors should be L2-normalised (length ≈ 1)."""
        embeddings = embedder.embed(["hello", "world", "foo bar baz"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_similar_texts_closer_than_dissimilar(self, embedder):
        anchor = embedder.embed_one("The cat sat on the mat")
        similar = embedder.embed_one("A cat rests on a rug")
        dissimilar = embedder.embed_one("Quantum mechanics describes subatomic particles")
        assert float(np.dot(anchor, similar)) > float(np.dot(anchor, dissimilar))

    def test_identical_texts_have_score_near_1(self, embedder):
        text = "machine learning is fascinating"
        e1 = embedder.embed_one(text)
        e2 = embedder.embed_one(text)
        assert float(np.dot(e1, e2)) > 0.999

    def test_embed_order_preserved(self, embedder):
        """embed(['a', 'b']) row 0 should match embed_one('a')."""
        batch = embedder.embed(["apple", "banana"])
        single = embedder.embed_one("apple")
        np.testing.assert_allclose(batch[0], single, atol=1e-5)

    def test_batch_and_single_consistent(self, embedder):
        texts = ["gradient descent", "neural network", "backpropagation"]
        batch = embedder.embed(texts)
        for i, text in enumerate(texts):
            single = embedder.embed_one(text)
            np.testing.assert_allclose(batch[i], single, atol=1e-5)
