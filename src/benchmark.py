"""
Benchmarking module — measures retrieval quality and ingestion speed.

Retrieval quality metrics (all computed at a given k):
  Precision@k  — of the top-k results, what fraction are relevant?
  Recall@k     — of all relevant chunks in the index, what fraction appear in top-k?
  MRR          — Mean Reciprocal Rank: 1/rank of the first relevant hit (0 if none)
  Hit Rate@k   — fraction of queries that have at least one relevant result in top-k

These are standard information-retrieval metrics. Together they give a picture
of both accuracy (precision) and coverage (recall), and MRR captures whether
the best result is ranked first.
"""

import time
from dataclasses import dataclass

from src.chunker import Chunk, chunk_by_sentence, chunk_text
from src.embedder import Embedder
from src.retriever import Retriever
from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSample:
    """
    A single evaluation query with its relevance criteria.

    A retrieved chunk is considered relevant if its text contains at least
    one keyword from relevant_keywords (case-insensitive substring match).
    Using keywords rather than exact passages makes evaluation robust to
    chunking boundaries — a chunk is relevant if it talks about the right
    topic, regardless of exactly where it was split.
    """

    query: str
    relevant_keywords: list[str]


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval quality metrics across a set of queries."""

    precision_at_k: float
    recall_at_k: float
    mrr: float        # Mean Reciprocal Rank
    hit_rate: float   # fraction of queries with ≥1 relevant result in top-k
    k: int            # the k value used for this evaluation


@dataclass
class IngestTiming:
    """Wall-clock timing for each stage of the ingestion pipeline (ms)."""

    chunk_ms: float
    embed_ms: float
    index_ms: float
    total_ms: float
    num_chunks: int


# ---------------------------------------------------------------------------
# Relevance helpers
# ---------------------------------------------------------------------------

def is_relevant(chunk: Chunk, sample: BenchmarkSample) -> bool:
    """
    Return True if the chunk is relevant to the benchmark sample.

    Args:
        chunk: A retrieved or indexed Chunk.
        sample: The query with its list of relevant keywords.

    Returns:
        True if any keyword appears in the chunk text (case-insensitive).
    """
    text_lower = chunk.text.lower()
    return any(kw.lower() in text_lower for kw in sample.relevant_keywords)


# ---------------------------------------------------------------------------
# Per-query metric functions
# ---------------------------------------------------------------------------

def precision_at_k(
    results: list[tuple[Chunk, float]],
    sample: BenchmarkSample,
    k: int,
) -> float:
    """
    Fraction of the top-k retrieved chunks that are relevant.

    Args:
        results: Ranked list of (Chunk, score) tuples.
        sample: The query being evaluated.
        k: Cut-off depth.

    Returns:
        Float in [0, 1]. Returns 0.0 if results is empty.
    """
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for chunk, _ in top_k if is_relevant(chunk, sample))
    return relevant / len(top_k)


def recall_at_k(
    results: list[tuple[Chunk, float]],
    sample: BenchmarkSample,
    k: int,
    total_relevant: int,
) -> float:
    """
    Fraction of all relevant chunks in the index that appear in top-k.

    Args:
        results: Ranked list of (Chunk, score) tuples.
        sample: The query being evaluated.
        k: Cut-off depth.
        total_relevant: Total relevant chunks in the entire index
                        (used as the denominator).

    Returns:
        Float in [0, 1]. Returns 0.0 if total_relevant is 0.
    """
    if total_relevant == 0:
        return 0.0
    top_k = results[:k]
    retrieved_relevant = sum(1 for chunk, _ in top_k if is_relevant(chunk, sample))
    return retrieved_relevant / total_relevant


def reciprocal_rank(
    results: list[tuple[Chunk, float]],
    sample: BenchmarkSample,
) -> float:
    """
    Reciprocal rank of the first relevant result.

    Returns 1/rank if a relevant result is found, 0.0 otherwise.
    Rank is 1-indexed: the top result has rank 1.

    Args:
        results: Ranked list of (Chunk, score) tuples.
        sample: The query being evaluated.

    Returns:
        Float in [0, 1].
    """
    for rank, (chunk, _) in enumerate(results, start=1):
        if is_relevant(chunk, sample):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    retriever: Retriever,
    samples: list[BenchmarkSample],
    k: int = 5,
) -> RetrievalMetrics:
    """
    Compute macro-averaged retrieval metrics across all benchmark samples.

    For each sample:
    1. Retrieve top-k chunks via the retriever
    2. Count how many relevant chunks exist in the full index (recall denominator)
    3. Compute per-query precision, recall, reciprocal rank, and hit

    Metrics are then averaged across all queries (macro-average).

    Args:
        retriever: A populated Retriever instance.
        samples: Evaluation queries with relevance keywords.
        k: Number of results to retrieve per query.

    Returns:
        RetrievalMetrics with macro-averaged scores.

    Raises:
        ValueError: If samples is empty.
    """
    if not samples:
        raise ValueError("samples must not be empty")

    precisions, recalls, rrs, hits = [], [], [], []

    for sample in samples:
        results = retriever.retrieve(sample.query, k=k, min_score=0.0)

        # Count relevant chunks across the entire index for recall denominator
        total_relevant = sum(
            1 for c in retriever.store.chunks if is_relevant(c, sample)
        )

        precisions.append(precision_at_k(results, sample, k))
        recalls.append(recall_at_k(results, sample, k, total_relevant))
        rrs.append(reciprocal_rank(results, sample))
        hits.append(1.0 if any(is_relevant(c, sample) for c, _ in results) else 0.0)

    n = len(samples)
    return RetrievalMetrics(
        precision_at_k=sum(precisions) / n,
        recall_at_k=sum(recalls) / n,
        mrr=sum(rrs) / n,
        hit_rate=sum(hits) / n,
        k=k,
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_ingest(
    text: str,
    embedder: Embedder,
    chunk_size: int = 512,
    overlap: int = 64,
) -> IngestTiming:
    """
    Measure wall-clock time (ms) for each stage of fixed-size ingestion.

    Stages timed independently:
      1. chunk_text — splitting the document
      2. embedder.embed — computing embeddings
      3. VectorStore.add — loading into FAISS

    Args:
        text: Pre-cleaned document text.
        embedder: An initialised Embedder.
        chunk_size: Characters per chunk.
        overlap: Overlap characters between adjacent chunks.

    Returns:
        IngestTiming with per-stage and total milliseconds.
    """
    t0 = time.perf_counter()
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    t1 = time.perf_counter()
    embeddings = embedder.embed([c.text for c in chunks])
    t2 = time.perf_counter()
    store = VectorStore(embedder.dimension)
    store.add(chunks, embeddings)
    t3 = time.perf_counter()

    def ms(a: float, b: float) -> float:
        return (b - a) * 1000

    return IngestTiming(
        chunk_ms=ms(t0, t1),
        embed_ms=ms(t1, t2),
        index_ms=ms(t2, t3),
        total_ms=ms(t0, t3),
        num_chunks=len(chunks),
    )


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

def compare_chunking_strategies(
    corpus: str,
    embedder: Embedder,
    samples: list[BenchmarkSample],
    k: int = 5,
) -> dict[str, RetrievalMetrics]:
    """
    Evaluate retrieval quality for both chunking strategies on the same corpus.

    Both strategies use the same embedder and evaluation samples, so any
    difference in metrics is attributable solely to the chunking approach.

    Args:
        corpus: Pre-cleaned document text to index.
        embedder: An initialised Embedder.
        samples: Evaluation queries with relevance keywords.
        k: Number of results to retrieve per query.

    Returns:
        Dict mapping strategy name ("fixed_size", "sentence_boundary")
        to its RetrievalMetrics.
    """
    strategies = {
        "fixed_size": chunk_text(corpus, chunk_size=512, overlap=64),
        "sentence_boundary": chunk_by_sentence(corpus, max_chunk_size=512),
    }

    results: dict[str, RetrievalMetrics] = {}
    for name, chunks in strategies.items():
        embeddings = embedder.embed([c.text for c in chunks])
        store = VectorStore(embedder.dimension)
        store.add(chunks, embeddings)
        retriever = Retriever(embedder, store)
        results[name] = evaluate_retrieval(retriever, samples, k=k)

    return results
