"""
Tests for the benchmarking module (src/benchmark.py).

Unit tests use hand-crafted Chunk lists and mock Retrievers — no model needed.
Integration tests use the session-scoped embedder from conftest.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.benchmark import (
    BenchmarkSample,
    IngestTiming,
    RetrievalMetrics,
    compare_chunking_strategies,
    evaluate_retrieval,
    is_relevant,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    time_ingest,
)
from src.chunker import Chunk
from src.retriever import Retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, index=idx, start_char=0,
                 end_char=len(text), metadata={})


def make_results(*texts: str) -> list[tuple[Chunk, float]]:
    return [(make_chunk(t, i), round(1.0 - i * 0.1, 2)) for i, t in enumerate(texts)]


def make_sample(query: str, *keywords: str) -> BenchmarkSample:
    return BenchmarkSample(query=query, relevant_keywords=list(keywords))


def mock_retriever(results: list[tuple[Chunk, float]],
                   all_chunks: list[Chunk] | None = None) -> Retriever:
    r = MagicMock(spec=Retriever)
    r.retrieve.return_value = results
    store = MagicMock()
    store.chunks = all_chunks if all_chunks is not None else [c for c, _ in results]
    r.store = store
    return r


# ---------------------------------------------------------------------------
# is_relevant
# ---------------------------------------------------------------------------

class TestIsRelevant:
    def test_matching_keyword(self):
        assert is_relevant(make_chunk("Gradient descent optimises neural networks."),
                           make_sample("q", "gradient descent")) is True

    def test_case_insensitive(self):
        assert is_relevant(make_chunk("GRADIENT DESCENT is used in training."),
                           make_sample("q", "gradient descent")) is True

    def test_no_matching_keyword(self):
        assert is_relevant(make_chunk("The French Revolution began in 1789."),
                           make_sample("q", "neural network", "loss function")) is False

    def test_any_keyword_sufficient(self):
        assert is_relevant(make_chunk("Fossil fuels release carbon dioxide."),
                           make_sample("q", "solar panels", "fossil fuel")) is True

    def test_partial_match(self):
        assert is_relevant(make_chunk("Backpropagation uses gradient information."),
                           make_sample("q", "gradient")) is True


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_relevant(self):
        sample = make_sample("q", "python")
        assert precision_at_k(make_results("Python is great.", "Python is fast.", "Python is fun."),
                              sample, k=3) == pytest.approx(1.0)

    def test_none_relevant(self):
        sample = make_sample("q", "neural network")
        assert precision_at_k(make_results("France.", "DNA structure.", "Climate change."),
                              sample, k=3) == pytest.approx(0.0)

    def test_half_relevant(self):
        sample = make_sample("q", "python")
        results = make_results("Python is great.", "The French Revolution.", "Python runs fast.")
        assert precision_at_k(results, sample, k=2) == pytest.approx(0.5)

    def test_k_larger_than_results(self):
        sample = make_sample("q", "python")
        assert precision_at_k(make_results("Python is great."), sample, k=5) == pytest.approx(1.0)

    def test_empty_results(self):
        assert precision_at_k([], make_sample("q", "python"), k=5) == pytest.approx(0.0)

    def test_k_cuts_off_correctly(self):
        sample = make_sample("q", "python")
        results = make_results("French Revolution.", "DNA helix.", "Python is great.")
        assert precision_at_k(results, sample, k=2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_full_recall(self):
        sample = make_sample("q", "python")
        results = make_results("Python is great.", "Python is fast.")
        assert recall_at_k(results, sample, k=5, total_relevant=2) == pytest.approx(1.0)

    def test_partial_recall(self):
        sample = make_sample("q", "python")
        results = make_results("Python is great.", "The French Revolution.", "Python runs fast.")
        assert recall_at_k(results, sample, k=1, total_relevant=2) == pytest.approx(0.5)

    def test_zero_recall(self):
        sample = make_sample("q", "python")
        assert recall_at_k(make_results("French Revolution.", "DNA helix."),
                           sample, k=5, total_relevant=3) == pytest.approx(0.0)

    def test_zero_total_relevant(self):
        sample = make_sample("q", "python")
        assert recall_at_k(make_results("Python is great."),
                           sample, k=5, total_relevant=0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------

class TestReciprocalRank:
    def test_first_rank(self):
        sample = make_sample("q", "python")
        assert reciprocal_rank(make_results("Python is great.", "French history."),
                               sample) == pytest.approx(1.0)

    def test_second_rank(self):
        sample = make_sample("q", "python")
        assert reciprocal_rank(make_results("French history.", "Python is great."),
                               sample) == pytest.approx(0.5)

    def test_third_rank(self):
        sample = make_sample("q", "python")
        assert reciprocal_rank(make_results("Climate change.", "DNA structure.", "Python is great."),
                               sample) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        sample = make_sample("q", "python")
        assert reciprocal_rank(make_results("Climate change.", "DNA structure."),
                               sample) == pytest.approx(0.0)

    def test_empty_results(self):
        assert reciprocal_rank([], make_sample("q", "python")) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate_retrieval
# ---------------------------------------------------------------------------

class TestEvaluateRetrieval:
    def test_returns_retrieval_metrics(self):
        chunk = make_chunk("Python is great.")
        metrics = evaluate_retrieval(mock_retriever([(chunk, 0.95)], [chunk]),
                                     [make_sample("q", "python")], k=1)
        assert isinstance(metrics, RetrievalMetrics)

    def test_perfect_retrieval(self):
        chunk = make_chunk("Python is the best language.")
        metrics = evaluate_retrieval(mock_retriever([(chunk, 0.95)], [chunk]),
                                     [make_sample("q", "python")], k=1)
        assert metrics.precision_at_k == pytest.approx(1.0)
        assert metrics.mrr == pytest.approx(1.0)
        assert metrics.hit_rate == pytest.approx(1.0)

    def test_zero_retrieval(self):
        chunk = make_chunk("The French Revolution.")
        metrics = evaluate_retrieval(mock_retriever([(chunk, 0.5)], [chunk]),
                                     [make_sample("q", "python")], k=5)
        assert metrics.precision_at_k == pytest.approx(0.0)
        assert metrics.mrr == pytest.approx(0.0)
        assert metrics.hit_rate == pytest.approx(0.0)

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError):
            evaluate_retrieval(mock_retriever([]), [], k=5)

    def test_k_stored_in_metrics(self):
        chunk = make_chunk("Python.")
        metrics = evaluate_retrieval(mock_retriever([(chunk, 0.9)], [chunk]),
                                     [make_sample("q", "python")], k=7)
        assert metrics.k == 7

    def test_macro_average_across_queries(self):
        s1 = make_sample("Tell me about python", "python")
        s2 = make_sample("Tell me about climate", "climate")
        hit_chunk = make_chunk("Python is great.")
        miss_chunk = make_chunk("The French Revolution.")

        def side_effect(query, k, min_score):
            return [(hit_chunk, 0.9)] if "python" in query.lower() else [(miss_chunk, 0.5)]

        retriever = MagicMock(spec=Retriever)
        retriever.retrieve.side_effect = side_effect
        retriever.store = MagicMock()
        retriever.store.chunks = [hit_chunk, miss_chunk]

        metrics = evaluate_retrieval(retriever, [s1, s2], k=1)
        assert metrics.precision_at_k == pytest.approx(0.5)
        assert metrics.hit_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# time_ingest
# ---------------------------------------------------------------------------

class TestTimeIngest:
    def test_returns_ingest_timing(self, embedder):
        timing = time_ingest("Machine learning uses gradient descent.", embedder)
        assert isinstance(timing, IngestTiming)

    def test_all_times_positive(self, embedder):
        timing = time_ingest("Python is a programming language. It is very popular.", embedder)
        assert timing.chunk_ms >= 0
        assert timing.embed_ms > 0
        assert timing.index_ms >= 0
        assert timing.total_ms > 0

    def test_total_is_sum_of_stages(self, embedder):
        timing = time_ingest("Climate change is caused by greenhouse gas emissions.", embedder)
        expected = timing.chunk_ms + timing.embed_ms + timing.index_ms
        assert timing.total_ms == pytest.approx(expected, rel=0.01)

    def test_num_chunks_positive(self, embedder):
        timing = time_ingest("DNA carries genetic information. Watson and Crick discovered its structure.", embedder)
        assert timing.num_chunks >= 1

    def test_larger_text_more_chunks(self, embedder):
        short = "One sentence."
        long = " ".join(["This is a sentence about machine learning."] * 20)
        t_short = time_ingest(short, embedder, chunk_size=100)
        t_long = time_ingest(long, embedder, chunk_size=100)
        assert t_long.num_chunks > t_short.num_chunks


# ---------------------------------------------------------------------------
# compare_chunking_strategies
# ---------------------------------------------------------------------------

MINI_CORPUS = (
    "Machine learning uses gradient descent to optimise neural networks. "
    "Climate change is driven by fossil fuel emissions and greenhouse gases. "
    "The French Revolution of 1789 ended the monarchy of France."
)
MINI_SAMPLES = [
    BenchmarkSample("neural networks", ["gradient descent", "neural"]),
    BenchmarkSample("global warming", ["greenhouse", "fossil fuel"]),
]


class TestCompareChunkingStrategies:
    def test_returns_both_strategies(self, embedder):
        result = compare_chunking_strategies(MINI_CORPUS, embedder, MINI_SAMPLES, k=3)
        assert "fixed_size" in result
        assert "sentence_boundary" in result

    def test_each_result_is_retrieval_metrics(self, embedder):
        result = compare_chunking_strategies(MINI_CORPUS, embedder, MINI_SAMPLES, k=3)
        for metrics in result.values():
            assert isinstance(metrics, RetrievalMetrics)

    def test_metrics_in_valid_range(self, embedder):
        result = compare_chunking_strategies(MINI_CORPUS, embedder, MINI_SAMPLES, k=3)
        for metrics in result.values():
            assert 0.0 <= metrics.precision_at_k <= 1.0
            assert 0.0 <= metrics.recall_at_k <= 1.0
            assert 0.0 <= metrics.mrr <= 1.0
            assert 0.0 <= metrics.hit_rate <= 1.0

    def test_k_propagated(self, embedder):
        result = compare_chunking_strategies(MINI_CORPUS, embedder, MINI_SAMPLES, k=2)
        for metrics in result.values():
            assert metrics.k == 2


# ---------------------------------------------------------------------------
# Integration — full benchmark corpus + real embedder
# ---------------------------------------------------------------------------

FULL_CORPUS = """\
Machine learning algorithms learn patterns from data without explicit programming. \
Gradient descent minimises a loss function during training of neural networks.

Climate change is caused by greenhouse gas emissions from burning fossil fuels. \
Carbon dioxide traps heat in the atmosphere, raising global temperatures.

The French Revolution of 1789 overthrew the monarchy. The storming of the \
Bastille became its defining symbol.

DNA carries genetic instructions in a double helix structure discovered by \
Watson and Crick. Mutations in genes can cause hereditary disease.

Python was created by Guido van Rossum. Its ecosystem including NumPy and \
PyTorch makes it dominant in data science and machine learning.\
"""

FULL_SAMPLES = [
    BenchmarkSample("How do neural networks learn?", ["gradient descent", "loss function"]),
    BenchmarkSample("What causes global warming?", ["greenhouse", "fossil fuel"]),
    BenchmarkSample("What is DNA?", ["dna", "double helix", "watson"]),
    BenchmarkSample("Why use Python for ML?", ["python", "numpy", "pytorch"]),
]


class TestBenchmarkIntegration:
    def test_hit_rate_is_high_for_good_corpus(self, embedder):
        result = compare_chunking_strategies(FULL_CORPUS, embedder, FULL_SAMPLES, k=3)
        for strategy, metrics in result.items():
            assert metrics.hit_rate >= 0.75, (
                f"{strategy}: hit_rate={metrics.hit_rate:.2f}"
            )

    def test_mrr_above_threshold(self, embedder):
        result = compare_chunking_strategies(FULL_CORPUS, embedder, FULL_SAMPLES, k=5)
        for strategy, metrics in result.items():
            assert metrics.mrr >= 0.5, f"{strategy}: MRR={metrics.mrr:.2f}"

    def test_time_ingest_completes(self, embedder):
        timing = time_ingest(FULL_CORPUS, embedder)
        assert timing.total_ms < 60_000
        assert timing.num_chunks >= 1
