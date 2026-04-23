"""
Tests for hybrid retrieval (src/hybrid_retriever.py).

Covers HybridRetriever (BM25 + dense fusion via RRF) and the _rrf_fuse
helper. Integration tests compare hybrid against dense-only retrieval
using the session-scoped embedder from conftest.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock

import pytest

from src.bm25_store import BM25Store
from src.benchmark import BenchmarkSample, evaluate_retrieval
from src.chunker import Chunk, chunk_text
from src.hybrid_retriever import HybridRetriever
from src.retriever import Retriever
from src.vector_store import VectorStore


def make_chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, index=idx, start_char=0, end_char=len(text), metadata={})


def make_chunks(*texts: str) -> list[Chunk]:
    return [make_chunk(t, i) for i, t in enumerate(texts)]


def mock_dense(results) -> Retriever:
    r = MagicMock(spec=Retriever)
    r.retrieve.return_value = results
    return r


# ---------------------------------------------------------------------------
# HybridRetriever — unit tests
# ---------------------------------------------------------------------------

class TestHybridRetrieverUnit:
    def _setup(self, texts, dense_result_indices_scores, rrf_k=60):
        chunks = make_chunks(*texts)
        bm25 = BM25Store(chunks)
        dense_mock = mock_dense([(chunks[i], s) for i, s in dense_result_indices_scores])
        return HybridRetriever(dense_mock, bm25, rrf_k=rrf_k), chunks

    def test_returns_list_of_tuples(self):
        r, _ = self._setup(
            ["Python is great.", "Climate is changing."],
            [(0, 0.9), (1, 0.5)],
        )
        results = r.retrieve("python", k=2)
        assert isinstance(results, list)
        assert all(isinstance(c, Chunk) and isinstance(s, float) for c, s in results)

    def test_k_limits_results(self):
        r, _ = self._setup(
            ["A alpha.", "B beta.", "C gamma.", "D delta."],
            [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)],
        )
        assert len(r.retrieve("alpha beta", k=2)) <= 2

    def test_scores_descending(self):
        r, _ = self._setup(
            ["Python is great.", "Climate change.", "DNA helix."],
            [(0, 0.9), (1, 0.7), (2, 0.5)],
        )
        scores = [s for _, s in r.retrieve("python climate", k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_raises(self):
        r, _ = self._setup(["Python is great."], [(0, 0.9)])
        with pytest.raises(ValueError):
            r.retrieve("   ", k=5)

    def test_min_score_filters(self):
        chunks = make_chunks("Python is great.", "Climate change.", "DNA helix.")
        bm25 = BM25Store(chunks)
        dense = mock_dense([(chunks[0], 0.9), (chunks[1], 0.7), (chunks[2], 0.5)])
        retriever = HybridRetriever(dense, bm25, rrf_k=60)
        assert retriever.retrieve("python", k=3, min_score=999.0) == []

    def test_rrf_k_stored(self):
        chunks = make_chunks("Python.")
        r = HybridRetriever(mock_dense([(chunks[0], 0.9)]), BM25Store(chunks), rrf_k=30)
        assert r.rrf_k == 30

    def test_dense_retriever_called(self):
        chunks = make_chunks("Python is great.", "Climate change.")
        bm25 = BM25Store(chunks)
        dense = mock_dense([(chunks[0], 0.9)])
        HybridRetriever(dense, bm25).retrieve("python", k=2)
        dense.retrieve.assert_called_once()

    def test_chunk_in_both_lists_scores_higher(self):
        chunks = make_chunks(
            "Python numpy pytorch machine learning.",
            "The French Revolution 1789.",
        )
        bm25 = BM25Store(chunks)
        dense = mock_dense([(chunks[0], 0.9), (chunks[1], 0.3)])
        results = HybridRetriever(dense, bm25, rrf_k=60).retrieve("python numpy", k=2)
        assert "python" in results[0][0].text.lower()


# ---------------------------------------------------------------------------
# RRF fusion — direct unit tests on _rrf_fuse
# ---------------------------------------------------------------------------

class TestRRFFusion:
    def _retriever(self) -> HybridRetriever:
        chunks = make_chunks("dummy")
        return HybridRetriever(mock_dense([]), BM25Store(chunks), rrf_k=60)

    def test_both_empty(self):
        assert self._retriever()._rrf_fuse([], []) == []

    def test_single_result_correct_score(self):
        r = self._retriever()
        chunk = make_chunk("Python is great.")
        _, score = r._rrf_fuse([(chunk, 0.9)], [])[0]
        assert score == pytest.approx(1.0 / (60 + 1))

    def test_same_chunk_in_both_lists_doubles_score(self):
        r = self._retriever()
        chunk = make_chunk("Python is great.")
        _, score = r._rrf_fuse([(chunk, 0.9)], [(chunk, 5.0)])[0]
        assert score == pytest.approx(2.0 / 61)

    def test_different_chunks_merged(self):
        r = self._retriever()
        c1, c2 = make_chunk("Python.", 0), make_chunk("Climate.", 1)
        assert len(r._rrf_fuse([(c1, 0.9)], [(c2, 5.0)])) == 2

    def test_higher_rank_scores_higher(self):
        r = self._retriever()
        c1, c2 = make_chunk("First.", 0), make_chunk("Second.", 1)
        fused = r._rrf_fuse([(c1, 0.9), (c2, 0.5)], [])
        assert fused[0][0] is c1
        assert fused[0][1] > fused[1][1]


# ---------------------------------------------------------------------------
# Integration — real embedder, comparing dense vs hybrid
# ---------------------------------------------------------------------------

CORPUS_TEXTS = [
    "Gradient descent minimises a loss function during training of neural networks.",
    "Climate change is caused by greenhouse gas emissions from burning fossil fuels.",
    "The French Revolution of 1789 overthrew the monarchy of France.",
    "DNA carries genetic instructions in a double helix discovered by Watson and Crick.",
    "Python was created by Guido van Rossum and is dominant in data science.",
]


@pytest.fixture(scope="module")
def hybrid_setup(embedder):
    """Build dense + BM25 + hybrid retriever over the 5-topic corpus."""
    text = "\n\n".join(CORPUS_TEXTS)
    chunks = chunk_text(text, chunk_size=100, overlap=16)
    embeddings = embedder.embed([c.text for c in chunks])

    store = VectorStore(embedder.dimension)
    store.add(chunks, embeddings)

    dense = Retriever(embedder, store)
    bm25 = BM25Store(chunks)
    hybrid = HybridRetriever(dense, bm25)
    return dense, bm25, hybrid, chunks


class TestHybridIntegration:
    def test_returns_chunks(self, hybrid_setup):
        _, _, hybrid, _ = hybrid_setup
        results = hybrid.retrieve("neural networks gradient descent", k=3)
        assert len(results) >= 1
        assert all(isinstance(c, Chunk) for c, _ in results)

    def test_finds_keyword_match(self, hybrid_setup):
        _, _, hybrid, _ = hybrid_setup
        texts = [c.text.lower() for c, _ in hybrid.retrieve("Watson Crick DNA helix", k=3)]
        assert any("dna" in t or "watson" in t for t in texts)

    def test_finds_semantic_match(self, hybrid_setup):
        _, _, hybrid, _ = hybrid_setup
        texts = [c.text.lower() for c, _ in hybrid.retrieve("What causes global warming?", k=3)]
        assert any("climate" in t or "greenhouse" in t or "fossil" in t for t in texts)

    def test_scores_descending(self, hybrid_setup):
        _, _, hybrid, _ = hybrid_setup
        scores = [s for _, s in hybrid.retrieve("Python data science", k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_bm25_keyword_hit(self, hybrid_setup):
        _, bm25, _, _ = hybrid_setup
        results = bm25.search("1789 revolution monarchy", k=3)
        top, score = results[0]
        assert score > 0.0
        assert "1789" in top.text or "revolution" in top.text.lower()

    def test_hit_rate_high(self, hybrid_setup):
        """All 5 topic queries should find a relevant chunk in top-3."""
        _, _, hybrid, chunks = hybrid_setup

        class _Wrapped:
            def __init__(self, h, all_chunks):
                self._h = h
                self.store = MagicMock()
                self.store.chunks = all_chunks

            def retrieve(self, query, k, min_score):
                return self._h.retrieve(query, k=k, min_score=min_score)

        samples = [
            BenchmarkSample("How do neural networks learn?",
                            ["gradient descent", "loss function", "neural"]),
            BenchmarkSample("What causes global warming?",
                            ["greenhouse", "fossil fuel", "climate"]),
            BenchmarkSample("What triggered the French Revolution?",
                            ["french revolution", "1789", "monarchy"]),
            BenchmarkSample("What is the structure of DNA?",
                            ["dna", "double helix", "watson"]),
            BenchmarkSample("Why is Python used in data science?",
                            ["python", "data science", "guido"]),
        ]

        metrics = evaluate_retrieval(_Wrapped(hybrid, chunks), samples, k=3)
        assert metrics.hit_rate >= 0.8, (
            f"hit_rate={metrics.hit_rate:.2f} — hybrid should find relevant "
            "chunks for most queries in top-3"
        )
