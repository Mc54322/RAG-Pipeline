"""
End-to-end evaluation framework for the RAG pipeline.

This module measures quality across the full pipeline — retrieval AND
generation — using structured QA pairs with reference answers.

It is distinct from src/benchmark.py, which measures retrieval quality
in isolation (no generator involved). The evaluator runs real (or mocked)
generation and scores answers against ground-truth references.

Answer quality metrics (SQuAD-style, no API key required):
  Exact Match (EM) — after normalisation (lowercase, strip punctuation
                     and articles), are the strings identical?
  Token F1         — token-level F1 between the bag of words of the
                     prediction and the reference. The standard metric
                     for extractive QA since SQuAD (Rajpurkar et al., 2016).

Context quality metric:
  Context Hit Rate — fraction of questions where at least one retrieved
                     chunk contains a keyword from relevant_keywords.
                     Measures whether retrieval surfaced the right material
                     for the generator to work with.
"""

import re
import string
from collections import Counter
from dataclasses import dataclass, field

from src.benchmark import is_relevant, BenchmarkSample
from src.chunker import Chunk
from src.generator import Generator
from src.prompt import build_prompt, get_system_prompt
from src.retriever import Retriever


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A single evaluation example: a question with a reference answer and
    keywords to check whether retrieval found relevant context.

    Args:
        question: The natural-language question to ask the pipeline.
        reference_answer: The ground-truth answer, used to score
                          the generated output.
        relevant_keywords: Substrings that should appear in a chunk for
                           it to be considered a context hit (same
                           semantics as BenchmarkSample.relevant_keywords).
    """

    question: str
    reference_answer: str
    relevant_keywords: list[str]


@dataclass
class QuestionResult:
    """
    Evaluation output for a single question.

    Args:
        question: The question asked.
        reference_answer: Ground-truth answer.
        generated_answer: Answer produced by the pipeline.
        retrieved_chunks: The (Chunk, score) pairs returned by the retriever.
        context_hit: True if at least one retrieved chunk is relevant.
        answer_f1: Token-level F1 of generated vs reference answer.
        answer_exact_match: True if normalised strings are identical.
    """

    question: str
    reference_answer: str
    generated_answer: str
    retrieved_chunks: list[tuple[Chunk, float]]
    context_hit: bool
    answer_f1: float
    answer_exact_match: bool


@dataclass
class EvaluationReport:
    """
    Aggregated evaluation metrics across a full dataset.

    Args:
        results: Per-question breakdown.
        context_hit_rate: Fraction of questions with at least one relevant
                          retrieved chunk (0–1).
        mean_answer_f1: Macro-average token F1 across all questions (0–1).
        exact_match_rate: Fraction of questions with exact-match answers (0–1).
        total_questions: Number of questions evaluated.
        k: Number of chunks retrieved per query.
    """

    results: list[QuestionResult] = field(default_factory=list)
    context_hit_rate: float = 0.0
    mean_answer_f1: float = 0.0
    exact_match_rate: float = 0.0
    total_questions: int = 0
    k: int = 5


# ---------------------------------------------------------------------------
# Answer normalisation and scoring (SQuAD metric)
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """
    Normalise an answer string for comparison.

    Applies the same preprocessing used by the SQuAD evaluation script:
      1. Lowercase
      2. Remove punctuation (replace with spaces)
      3. Remove English articles (a, an, the)
      4. Collapse whitespace

    Args:
        text: Raw answer string (predicted or reference).

    Returns:
        Normalised string suitable for exact-match or token-F1 comparison.
    """
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation,
                                        " " * len(string.punctuation)))
    # Remove standalone articles
    tokens = text.split()
    tokens = [t for t in tokens if t not in {"a", "an", "the"}]
    return " ".join(tokens)


def token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 between a predicted and reference answer.

    Both strings are normalised before comparison. F1 is the harmonic mean
    of token precision and recall over the bags of words.

    This is the primary answer-quality metric in SQuAD. It handles partial
    credit well: an answer that contains most of the right information but
    adds extra words scores better than one that is completely wrong.

    Args:
        prediction: The generated answer.
        reference: The ground-truth answer.

    Returns:
        Float in [0, 1]. Returns 0.0 if either string normalises to empty.
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # Number of tokens that appear in both bags
    common = sum((pred_counter & ref_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> bool:
    """
    Check whether two answers are identical after normalisation.

    Args:
        prediction: The generated answer.
        reference: The ground-truth answer.

    Returns:
        True if the normalised strings are equal.
    """
    return normalize_answer(prediction) == normalize_answer(reference)


# ---------------------------------------------------------------------------
# Pipeline evaluator
# ---------------------------------------------------------------------------

class PipelineEvaluator:
    """
    Runs the full RAG pipeline on a dataset and scores the output.

    Ties together retrieval, prompt construction, generation, and scoring
    in a single evaluate() call. The generator is injected so it can be
    mocked in tests (no real API calls).

    Args:
        retriever: A populated Retriever (or any object with .retrieve()).
        generator: A Generator instance (or mock with .generate()).
        k: Number of chunks to retrieve per query.
        min_score: Similarity threshold passed to the retriever.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.k = k
        self.min_score = min_score

    def evaluate(self, dataset: list[QAPair]) -> EvaluationReport:
        """
        Run the pipeline on every QAPair and return aggregated metrics.

        For each question:
          1. Retrieve top-k chunks
          2. Build a RAG prompt from the chunks
          3. Generate an answer
          4. Score: context hit, token F1, exact match

        Args:
            dataset: List of QAPair evaluation examples.

        Returns:
            EvaluationReport with per-question results and macro-averages.

        Raises:
            ValueError: If dataset is empty.
        """
        if not dataset:
            raise ValueError("dataset must not be empty")

        results: list[QuestionResult] = []
        system = get_system_prompt()

        for pair in dataset:
            chunks = self.retriever.retrieve(
                pair.question, k=self.k, min_score=self.min_score
            )

            # Build a BenchmarkSample-compatible object to reuse is_relevant
            sample = BenchmarkSample(
                query=pair.question,
                relevant_keywords=pair.relevant_keywords,
            )
            hit = any(is_relevant(chunk, sample) for chunk, _ in chunks)

            prompt = build_prompt(pair.question, chunks)
            answer = self.generator.generate(prompt, system=system)

            results.append(QuestionResult(
                question=pair.question,
                reference_answer=pair.reference_answer,
                generated_answer=answer,
                retrieved_chunks=chunks,
                context_hit=hit,
                answer_f1=token_f1(answer, pair.reference_answer),
                answer_exact_match=exact_match(answer, pair.reference_answer),
            ))

        n = len(results)
        return EvaluationReport(
            results=results,
            context_hit_rate=sum(r.context_hit for r in results) / n,
            mean_answer_f1=sum(r.answer_f1 for r in results) / n,
            exact_match_rate=sum(r.answer_exact_match for r in results) / n,
            total_questions=n,
            k=self.k,
        )
