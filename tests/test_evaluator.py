"""
Tests for the end-to-end evaluation framework (src/evaluator.py).

Unit tests mock both Retriever and Generator — no model needed.
Integration tests use the session-scoped embedder from conftest.py
with a mock Generator, exercising the full retrieval + prompt + score path.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.chunker import Chunk, chunk_text
from src.evaluator import (
    EvaluationReport,
    PipelineEvaluator,
    QAPair,
    QuestionResult,
    exact_match,
    normalize_answer,
    token_f1,
)
from src.retriever import Retriever
from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, index=idx, start_char=0,
                 end_char=len(text), metadata={})


def mock_retriever(chunks_and_scores: list[tuple[Chunk, float]]) -> Retriever:
    r = MagicMock(spec=Retriever)
    r.retrieve.return_value = chunks_and_scores
    return r


def mock_generator(answer: str = "Mocked answer.") -> MagicMock:
    g = MagicMock()
    g.generate.return_value = answer
    return g


def make_pair(
    question: str = "What is Python?",
    reference: str = "Python is a programming language.",
    keywords: list[str] | None = None,
) -> QAPair:
    return QAPair(question=question, reference_answer=reference,
                  relevant_keywords=keywords or ["python"])


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------

class TestNormalizeAnswer:
    def test_lowercases(self):
        assert normalize_answer("Python") == "python"

    def test_removes_punctuation(self):
        result = normalize_answer("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_removes_articles(self):
        result = normalize_answer("The cat sat on a mat")
        assert "the" not in result.split()
        assert "a" not in result.split()

    def test_removes_an(self):
        assert "an" not in normalize_answer("An apple a day").split()

    def test_collapses_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_only_articles(self):
        assert normalize_answer("a the an") == ""

    def test_preserves_content_words(self):
        result = normalize_answer("gradient descent")
        assert "gradient" in result
        assert "descent" in result


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------

class TestTokenF1:
    def test_identical_answers(self):
        assert token_f1("Python is great", "Python is great") == pytest.approx(1.0)

    def test_completely_different(self):
        assert token_f1("France revolution", "gradient descent neural") == pytest.approx(0.0)

    def test_partial_overlap(self):
        # "Python is great" vs "Python is fast" → 2 common tokens, F1 = 2/3
        assert token_f1("Python is great", "Python is fast") == pytest.approx(2 / 3)

    def test_empty_prediction(self):
        assert token_f1("", "Python is great") == pytest.approx(0.0)

    def test_empty_reference(self):
        assert token_f1("Python is great", "") == pytest.approx(0.0)

    def test_both_empty(self):
        assert token_f1("", "") == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert token_f1("PYTHON", "python") == pytest.approx(1.0)

    def test_articles_ignored(self):
        assert token_f1("the python language", "python language") == pytest.approx(1.0)

    def test_repeated_tokens_counted_once(self):
        f1 = token_f1("python python", "python")
        assert 0.0 < f1 < 1.0

    def test_symmetric(self):
        a, b = "machine learning gradient", "gradient descent machine"
        assert token_f1(a, b) == pytest.approx(token_f1(b, a))

    def test_subset_answer(self):
        # pred: [python]  ref: [python, data, science, numpy]
        # precision: 1/1  recall: 1/4  F1: 0.4
        assert token_f1("python", "python data science numpy") == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_identical(self):
        assert exact_match("Python is great", "Python is great") is True

    def test_case_insensitive(self):
        assert exact_match("PYTHON", "python") is True

    def test_articles_stripped(self):
        assert exact_match("the Python language", "Python language") is True

    def test_punctuation_stripped(self):
        assert exact_match("Python!", "Python") is True

    def test_different_content(self):
        assert exact_match("Python is great", "Climate change") is False

    def test_extra_words_no_match(self):
        assert exact_match("Python is a great language", "Python") is False

    def test_both_empty_after_normalisation(self):
        assert exact_match("the a an", "a the an") is True


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_qa_pair_fields(self):
        pair = QAPair("Q?", "A.", ["keyword"])
        assert pair.question == "Q?"
        assert pair.reference_answer == "A."
        assert pair.relevant_keywords == ["keyword"]

    def test_question_result_fields(self):
        result = QuestionResult(
            question="Q?", reference_answer="A.", generated_answer="B.",
            retrieved_chunks=[(make_chunk("Python text."), 0.8)],
            context_hit=True, answer_f1=0.5, answer_exact_match=False,
        )
        assert result.context_hit is True
        assert result.answer_f1 == pytest.approx(0.5)

    def test_evaluation_report_defaults(self):
        report = EvaluationReport()
        assert report.results == []
        assert report.total_questions == 0
        assert report.k == 5


# ---------------------------------------------------------------------------
# PipelineEvaluator — unit tests (mocked retriever + generator)
# ---------------------------------------------------------------------------

class TestPipelineEvaluatorUnit:
    def _make_evaluator(self, answer: str = "Python is a programming language.",
                        chunk_text_: str = "Python is a programming language.",
                        k: int = 3) -> tuple[PipelineEvaluator, Chunk]:
        chunk = make_chunk(chunk_text_)
        evaluator = PipelineEvaluator(mock_retriever([(chunk, 0.9)]),
                                      mock_generator(answer), k=k)
        return evaluator, chunk

    def test_returns_evaluation_report(self):
        evaluator, _ = self._make_evaluator()
        assert isinstance(evaluator.evaluate([make_pair()]), EvaluationReport)

    def test_empty_dataset_raises(self):
        evaluator, _ = self._make_evaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate([])

    def test_total_questions_correct(self):
        evaluator, _ = self._make_evaluator()
        report = evaluator.evaluate([make_pair("Q1?", "A1."), make_pair("Q2?", "A2.")])
        assert report.total_questions == 2

    def test_k_stored_in_report(self):
        evaluator, _ = self._make_evaluator(k=7)
        assert evaluator.evaluate([make_pair()]).k == 7

    def test_retriever_called_per_question(self):
        chunk = make_chunk("Python is a programming language.")
        retriever = mock_retriever([(chunk, 0.9)])
        evaluator = PipelineEvaluator(retriever, mock_generator(), k=3)
        evaluator.evaluate([make_pair("Q1?"), make_pair("Q2?")])
        assert retriever.retrieve.call_count == 2

    def test_generator_called_per_question(self):
        chunk = make_chunk("Python is a programming language.")
        generator = mock_generator()
        evaluator = PipelineEvaluator(mock_retriever([(chunk, 0.9)]), generator, k=3)
        evaluator.evaluate([make_pair("Q1?"), make_pair("Q2?")])
        assert generator.generate.call_count == 2

    def test_perfect_answer_scores_one(self):
        reference = "Python is a programming language"
        evaluator, _ = self._make_evaluator(answer=reference)
        report = evaluator.evaluate([make_pair(reference=reference)])
        assert report.mean_answer_f1 == pytest.approx(1.0)
        assert report.exact_match_rate == pytest.approx(1.0)

    def test_wrong_answer_scores_zero(self):
        evaluator, _ = self._make_evaluator(
            answer="The French Revolution began in 1789.",
            chunk_text_="Python is popular.",
        )
        report = evaluator.evaluate([make_pair(reference="Python is a programming language.")])
        assert report.mean_answer_f1 == pytest.approx(0.0)
        assert report.exact_match_rate == pytest.approx(0.0)

    def test_context_hit_true_when_relevant(self):
        evaluator, _ = self._make_evaluator(chunk_text_="Python is a programming language.")
        report = evaluator.evaluate([make_pair(keywords=["python"])])
        assert report.context_hit_rate == pytest.approx(1.0)

    def test_context_hit_false_when_irrelevant(self):
        evaluator, _ = self._make_evaluator(chunk_text_="The French Revolution began in 1789.")
        report = evaluator.evaluate([make_pair(keywords=["python", "numpy"])])
        assert report.context_hit_rate == pytest.approx(0.0)

    def test_macro_average_f1(self):
        """Two questions: one perfect F1, one zero — average should be 0.5."""
        chunk = make_chunk("Python is popular. Climate change is real.")
        retriever = MagicMock(spec=Retriever)
        retriever.retrieve.return_value = [(chunk, 0.9)]

        answers = ["python programming language", "wrong answer here"]
        call_count = 0

        def gen_side_effect(prompt, system=""):
            nonlocal call_count
            answer = answers[call_count]
            call_count += 1
            return answer

        generator = MagicMock()
        generator.generate.side_effect = gen_side_effect

        evaluator = PipelineEvaluator(retriever, generator, k=3)
        report = evaluator.evaluate([
            QAPair("Q1?", "python programming language", ["python"]),
            QAPair("Q2?", "completely different content", ["climate"]),
        ])
        assert report.mean_answer_f1 == pytest.approx(0.5, abs=0.01)

    def test_results_list_length(self):
        evaluator, _ = self._make_evaluator()
        assert len(evaluator.evaluate([make_pair(), make_pair(), make_pair()]).results) == 3

    def test_result_contains_generated_answer(self):
        evaluator, _ = self._make_evaluator(answer="My generated answer.")
        assert evaluator.evaluate([make_pair()]).results[0].generated_answer == "My generated answer."

    def test_result_contains_retrieved_chunks(self):
        chunk = make_chunk("Python is popular.")
        evaluator = PipelineEvaluator(mock_retriever([(chunk, 0.85)]), mock_generator())
        report = evaluator.evaluate([make_pair()])
        assert report.results[0].retrieved_chunks[0][0] is chunk


# ---------------------------------------------------------------------------
# Integration — real Embedder, mock Generator
# ---------------------------------------------------------------------------

EVAL_CORPUS = [
    "Gradient descent is the primary algorithm for training neural networks. "
    "It minimises a loss function by iteratively adjusting model weights.",
    "Climate change is caused by greenhouse gas emissions from burning fossil fuels. "
    "Carbon dioxide traps heat in the atmosphere.",
    "DNA is a double helix structure discovered by Watson and Crick in 1953. "
    "It carries genetic instructions for all living organisms.",
    "Python was created by Guido van Rossum. "
    "Libraries like NumPy and PyTorch make it dominant in machine learning.",
]

EVAL_DATASET = [
    QAPair("How do neural networks learn?",
           "Neural networks learn using gradient descent to minimise a loss function.",
           ["gradient descent", "loss function", "neural"]),
    QAPair("What causes global warming?",
           "Greenhouse gas emissions from burning fossil fuels cause global warming.",
           ["greenhouse", "fossil fuel", "carbon dioxide"]),
    QAPair("What is DNA?",
           "DNA is a double helix structure carrying genetic instructions.",
           ["dna", "double helix", "watson"]),
    QAPair("Why is Python popular for machine learning?",
           "Python is popular for machine learning due to libraries like NumPy and PyTorch.",
           ["python", "numpy", "pytorch"]),
]


@pytest.fixture(scope="module")
def integration_evaluator(embedder):
    """PipelineEvaluator with real Embedder and mock Generator."""
    text = "\n\n".join(EVAL_CORPUS)
    chunks = chunk_text(text, chunk_size=256, overlap=32)
    embeddings = embedder.embed([c.text for c in chunks])

    store = VectorStore(embedder.dimension)
    store.add(chunks, embeddings)
    retriever = Retriever(embedder, store)

    def smart_generate(prompt, system=""):
        for pair in EVAL_DATASET:
            if pair.question.lower()[:20] in prompt.lower():
                return pair.reference_answer
        return "I don't know."

    generator = MagicMock()
    generator.generate.side_effect = smart_generate

    return PipelineEvaluator(retriever, generator, k=3)


class TestEvaluatorIntegration:
    def test_returns_report(self, integration_evaluator):
        assert isinstance(integration_evaluator.evaluate(EVAL_DATASET), EvaluationReport)

    def test_correct_question_count(self, integration_evaluator):
        report = integration_evaluator.evaluate(EVAL_DATASET)
        assert report.total_questions == len(EVAL_DATASET)
        assert len(report.results) == len(EVAL_DATASET)

    def test_context_hit_rate_high(self, integration_evaluator):
        report = integration_evaluator.evaluate(EVAL_DATASET)
        assert report.context_hit_rate >= 0.75, (
            f"context_hit_rate={report.context_hit_rate:.2f}"
        )

    def test_answer_f1_high_when_generator_is_perfect(self, integration_evaluator):
        report = integration_evaluator.evaluate(EVAL_DATASET)
        assert report.mean_answer_f1 >= 0.5, (
            f"mean_f1={report.mean_answer_f1:.2f}"
        )

    def test_all_results_have_chunks(self, integration_evaluator):
        report = integration_evaluator.evaluate(EVAL_DATASET)
        for result in report.results:
            assert len(result.retrieved_chunks) >= 1

    def test_scores_in_valid_range(self, integration_evaluator):
        report = integration_evaluator.evaluate(EVAL_DATASET)
        assert 0.0 <= report.context_hit_rate <= 1.0
        assert 0.0 <= report.mean_answer_f1 <= 1.0
        assert 0.0 <= report.exact_match_rate <= 1.0
