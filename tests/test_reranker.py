"""
Tests for cross-encoder re-ranking (src/reranker.py).

Unit tests mock the CrossEncoder so no model is loaded — fast and offline.
Integration tests use the session-scoped reranker fixture from conftest.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.chunker import Chunk
from src.reranker import Reranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, index=idx, start_char=0,
                 end_char=len(text), metadata={})


def make_candidates(*texts: str) -> list[tuple[Chunk, float]]:
    """Build a (Chunk, dummy_score) list from plain strings."""
    return [(make_chunk(t, i), 0.5) for i, t in enumerate(texts)]


def mock_reranker(scores: list[float]) -> Reranker:
    """
    Return a Reranker with a mocked CrossEncoder that returns fixed scores.

    Scores are returned in the same order as the input pairs, so the test
    controls which candidate ends up ranked first.
    """
    with patch("src.reranker.CrossEncoder") as MockCE:
        instance = MagicMock()
        instance.predict.return_value = np.array(scores, dtype=np.float32)
        MockCE.return_value = instance
        r = Reranker()
    r._model = instance
    return r


# ---------------------------------------------------------------------------
# Unit tests — mocked CrossEncoder
# ---------------------------------------------------------------------------

class TestRerankerInit:
    def test_default_model_name(self):
        with patch("src.reranker.CrossEncoder"):
            r = Reranker()
        assert r.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_custom_model_name(self):
        with patch("src.reranker.CrossEncoder"):
            r = Reranker(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
        assert r.model_name == "cross-encoder/ms-marco-TinyBERT-L-2-v2"

    def test_cross_encoder_constructed(self):
        with patch("src.reranker.CrossEncoder") as MockCE:
            Reranker()
        MockCE.assert_called_once()


class TestRerankerRerank:
    def test_empty_candidates_returns_empty(self):
        r = mock_reranker([])
        assert r.rerank("What is Python?", []) == []

    def test_empty_query_raises(self):
        r = mock_reranker([1.0])
        with pytest.raises(ValueError):
            r.rerank("   ", make_candidates("Python is great."))

    def test_returns_list_of_tuples(self):
        r = mock_reranker([0.8, 0.3])
        results = r.rerank("Python", make_candidates("Python is great.", "Climate change."))
        assert isinstance(results, list)
        assert all(isinstance(c, Chunk) and isinstance(s, float) for c, s in results)

    def test_sorted_by_score_descending(self):
        # Give second candidate the higher score → it should come first
        r = mock_reranker([0.2, 0.9])
        candidates = make_candidates("French Revolution.", "Python is popular.")
        results = r.rerank("Python programming", candidates)
        assert results[0][0].text == "Python is popular."
        assert results[1][0].text == "French Revolution."

    def test_scores_replaced_by_cross_encoder(self):
        """The first-stage scores (0.5) should be replaced by CE scores."""
        r = mock_reranker([2.5, -1.0])
        results = r.rerank("query", make_candidates("A text.", "B text."))
        scores = [s for _, s in results]
        assert scores[0] == pytest.approx(2.5)
        assert scores[1] == pytest.approx(-1.0)

    def test_scores_can_be_negative(self):
        """Cross-encoder logits are unbounded — negatives are valid."""
        r = mock_reranker([-5.0, -3.0, -8.0])
        results = r.rerank("query", make_candidates("A.", "B.", "C."))
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(-3.0)

    def test_top_n_limits_results(self):
        r = mock_reranker([0.9, 0.7, 0.5, 0.3])
        results = r.rerank("query", make_candidates("A.", "B.", "C.", "D."), top_n=2)
        assert len(results) == 2

    def test_top_n_none_returns_all(self):
        r = mock_reranker([0.9, 0.7, 0.5])
        results = r.rerank("query", make_candidates("A.", "B.", "C."), top_n=None)
        assert len(results) == 3

    def test_top_n_larger_than_candidates(self):
        r = mock_reranker([0.9, 0.5])
        results = r.rerank("query", make_candidates("A.", "B."), top_n=10)
        assert len(results) == 2

    def test_single_candidate_returned(self):
        r = mock_reranker([1.5])
        results = r.rerank("Python", make_candidates("Python is great."))
        assert len(results) == 1
        assert results[0][1] == pytest.approx(1.5)

    def test_predict_called_with_correct_pairs(self):
        r = mock_reranker([0.5, 0.3])
        chunk_a = make_chunk("Python is popular.")
        chunk_b = make_chunk("Climate change is real.")
        r.rerank("Python programming", [(chunk_a, 0.9), (chunk_b, 0.7)])
        call_args = r._model.predict.call_args[0][0]
        assert call_args[0] == ("Python programming", "Python is popular.")
        assert call_args[1] == ("Python programming", "Climate change is real.")

    def test_chunk_objects_preserved(self):
        """Results must contain the same Chunk objects, not copies."""
        r = mock_reranker([0.3, 0.9])
        chunk_a = make_chunk("Python is popular.", 0)
        chunk_b = make_chunk("Climate change is real.", 1)
        results = r.rerank("Python", [(chunk_a, 0.9), (chunk_b, 0.5)])
        result_chunks = [c for c, _ in results]
        assert result_chunks[0] is chunk_b
        assert result_chunks[1] is chunk_a

    def test_all_same_score_order_stable(self):
        """When all scores equal, all candidates should still be returned."""
        r = mock_reranker([0.5, 0.5, 0.5])
        results = r.rerank("query", make_candidates("A.", "B.", "C."))
        assert len(results) == 3

    def test_rerank_reverses_bad_first_stage(self):
        """Worst candidate first-stage gets best cross-encoder score."""
        r = mock_reranker([0.1, 0.2, 0.9])
        candidates = make_candidates("Irrelevant A.", "Irrelevant B.", "Highly relevant.")
        results = r.rerank("query", candidates)
        assert results[0][0].text == "Highly relevant."


# ---------------------------------------------------------------------------
# Integration tests — real cross-encoder model (reranker from conftest.py)
# ---------------------------------------------------------------------------

RELEVANT_TEXTS = [
    "Gradient descent minimises a loss function by iteratively adjusting "
    "model weights to reduce prediction error during neural network training.",
    "The French Revolution of 1789 overthrew the monarchy of France after "
    "years of fiscal crisis and social inequality.",
    "DNA carries genetic instructions in a double helix structure described "
    "by Watson and Crick in 1953.",
    "Python was created by Guido van Rossum and is dominant in data science "
    "thanks to libraries like NumPy and PyTorch.",
    "Climate change is driven by greenhouse gas emissions from burning "
    "fossil fuels, trapping heat in the atmosphere.",
]


class TestRerankerIntegration:
    def test_relevant_ranked_above_irrelevant(self, reranker):
        """The on-topic passage should score higher than off-topic ones."""
        query = "How do neural networks learn?"
        relevant = make_chunk(RELEVANT_TEXTS[0])
        irrelevant_a = make_chunk(RELEVANT_TEXTS[1])
        irrelevant_b = make_chunk(RELEVANT_TEXTS[4])
        # Deliberately put relevant last so first-stage order is wrong
        candidates = [(irrelevant_a, 0.9), (irrelevant_b, 0.8), (relevant, 0.3)]
        results = reranker.rerank(query, candidates)
        assert results[0][0] is relevant

    def test_scores_change_with_query(self, reranker):
        """Different queries produce different score orderings."""
        chunk_a = make_chunk(RELEVANT_TEXTS[0])  # neural networks
        chunk_b = make_chunk(RELEVANT_TEXTS[2])  # DNA
        candidates = [(chunk_a, 0.7), (chunk_b, 0.7)]

        results_nn = reranker.rerank("How do neural networks learn?", candidates)
        results_dna = reranker.rerank("What is the structure of DNA?", candidates)

        assert results_nn[0][0] is chunk_a
        assert results_dna[0][0] is chunk_b

    def test_top_n_integration(self, reranker):
        chunks = [make_chunk(t, i) for i, t in enumerate(RELEVANT_TEXTS)]
        results = reranker.rerank("Python machine learning",
                                  [(c, 0.5) for c in chunks], top_n=2)
        assert len(results) == 2

    def test_scores_are_floats(self, reranker):
        results = reranker.rerank("neural networks",
                                  make_candidates(RELEVANT_TEXTS[0], RELEVANT_TEXTS[1]))
        for _, score in results:
            assert isinstance(score, float)

    def test_output_sorted_descending(self, reranker):
        chunks = [make_chunk(t, i) for i, t in enumerate(RELEVANT_TEXTS)]
        results = reranker.rerank("climate change fossil fuels",
                                  [(c, 0.5) for c in chunks])
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_python_chunk_wins_python_query(self, reranker):
        query = "Why is Python popular for machine learning?"
        python_chunk = make_chunk(RELEVANT_TEXTS[3])
        other_chunks = [make_chunk(t, i + 1) for i, t in enumerate(RELEVANT_TEXTS[:3])]
        candidates = [(c, 0.6) for c in other_chunks] + [(python_chunk, 0.4)]
        results = reranker.rerank(query, candidates)
        assert results[0][0] is python_chunk

    def test_empty_candidates_integration(self, reranker):
        assert reranker.rerank("Python", []) == []
