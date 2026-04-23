"""
RAG Pipeline — Retrieval Benchmark Runner

Evaluates retrieval quality and ingestion speed across both chunking
strategies using a five-topic corpus loaded from the benchmark PDFs in
data/benchmark/. Generate those PDFs first if they don't exist:

    python scripts/create_benchmark_data.py
    python scripts/run_benchmark.py

No API key required — benchmarks only test the embedder + FAISS retrieval
stack, not the LLM generation step.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import (
    BenchmarkSample,
    compare_chunking_strategies,
    evaluate_retrieval,
    time_ingest,
)
from src.bm25_store import BM25Store
from src.chunker import chunk_text
from src.embedder import Embedder
from src.hybrid_retriever import HybridRetriever
from src.ingestion import clean_text, load_pdf
from src.reranker import Reranker
from src.retriever import Retriever
from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Benchmark PDF directory
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).parent.parent / "data" / "benchmark"

PDF_FILES = [
    "machine_learning.pdf",
    "climate_change.pdf",
    "french_revolution.pdf",
    "dna_genetics.pdf",
    "python_programming.pdf",
]


def load_corpus() -> str:
    """
    Load and concatenate text from all benchmark PDFs.

    Each PDF contributes one topic paragraph block. The combined string
    is used as the single corpus for both chunking strategies so that
    results across strategies are directly comparable.

    Returns:
        Clean concatenated text from all benchmark PDFs.

    Raises:
        FileNotFoundError: If any benchmark PDF is missing. Run
            scripts/create_benchmark_data.py to generate them.
    """
    parts: list[str] = []
    for filename in PDF_FILES:
        path = BENCHMARK_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark PDF not found: {path}\n"
                "Run: python scripts/create_benchmark_data.py"
            )
        raw = load_pdf(path)
        parts.append(clean_text(raw))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Evaluation samples
# Queries are deliberate paraphrases so the model must rely on semantic
# understanding, not keyword overlap with the source text.
# ---------------------------------------------------------------------------

SAMPLES: list[BenchmarkSample] = [
    BenchmarkSample(
        query="How do neural networks learn from data?",
        relevant_keywords=["gradient descent", "neural network", "loss function"],
    ),
    BenchmarkSample(
        query="What is responsible for global warming?",
        relevant_keywords=["greenhouse", "fossil fuel", "carbon dioxide", "climate"],
    ),
    BenchmarkSample(
        query="What triggered the French Revolution?",
        relevant_keywords=["french revolution", "bastille", "monarchy", "1789"],
    ),
    BenchmarkSample(
        query="What is the structure of DNA?",
        relevant_keywords=["dna", "double helix", "watson", "crick", "genetic"],
    ),
    BenchmarkSample(
        query="Why is Python popular for machine learning?",
        relevant_keywords=["python", "numpy", "pytorch", "data science"],
    ),
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

W = 66  # total table width


def hr(char: str = "─") -> str:
    return char * W


def section(title: str) -> str:
    return f"\n── {title} {'─' * (W - len(title) - 4)}"


def metrics_table(
    rows: dict[str, object],
    k: int,
) -> str:
    col_w = 13
    header = (
        f"  {'Strategy':<20}"
        f"{'P@' + str(k):>{col_w}}"
        f"{'R@' + str(k):>{col_w}}"
        f"{'MRR':>{col_w}}"
        f"{'Hit Rate':>{col_w}}"
    )
    divider = "  " + "-" * (W - 2)
    lines = [header, divider]
    for name, m in rows.items():
        label = name.replace("_", " ").title()
        lines.append(
            f"  {label:<20}"
            f"{m.precision_at_k:>{col_w}.3f}"
            f"{m.recall_at_k:>{col_w}.3f}"
            f"{m.mrr:>{col_w}.3f}"
            f"{m.hit_rate:>{col_w}.3f}"
        )
    return "\n".join(lines)


def timing_block(label: str, timing: object) -> str:
    return (
        f"  {label}\n"
        f"    Chunks created : {timing.num_chunks}\n"
        f"    Chunking       : {timing.chunk_ms:6.1f} ms\n"
        f"    Embedding      : {timing.embed_ms:6.1f} ms\n"
        f"    FAISS indexing : {timing.index_ms:6.1f} ms\n"
        f"    Total          : {timing.total_ms:6.1f} ms"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * W)
    print(" RAG Pipeline — Retrieval Benchmark")
    print("=" * W)

    # ── Load corpus from PDFs ────────────────────────────────────────────────
    print(f"\nLoading corpus from {BENCHMARK_DIR}/")
    corpus = load_corpus()
    word_count = len(corpus.split())
    print(f"  PDFs loaded : {len(PDF_FILES)}")
    print(f"  Total words : {word_count}")

    # ── Load embedding model ─────────────────────────────────────────────────
    print("\nLoading embedding model…", flush=True)
    embedder = Embedder()
    print(f"  Model   : {embedder.model_name}")
    print(f"  Dim     : {embedder.dimension}")
    print(f"  Queries : {len(SAMPLES)} evaluation samples")

    # ── Build fixed-size index (shared across sections) ──────────────────────
    chunks = chunk_text(corpus, chunk_size=512, overlap=64)
    embeddings = embedder.embed([c.text for c in chunks])
    store = VectorStore(embedder.dimension)
    store.add(chunks, embeddings)
    retriever = Retriever(embedder, store)
    bm25_store = BM25Store(chunks)
    hybrid = HybridRetriever(retriever, bm25_store)

    # evaluate_retrieval expects a retriever-like object with .store.chunks
    # Wrap HybridRetriever so it fits the same interface
    class _HybridAdapter:
        def __init__(self, h: HybridRetriever, all_chunks: list) -> None:
            self._h = h
            self.store = type("S", (), {"chunks": all_chunks})()

        def retrieve(self, query: str, k: int, min_score: float) -> list:
            return self._h.retrieve(query, k=k, min_score=min_score)

    hybrid_adapter = _HybridAdapter(hybrid, chunks)

    # Wrap dense+reranker so evaluate_retrieval can call it.
    # Fetches up to fetch_k candidates from dense, then re-ranks to k.
    class _RerankerAdapter:
        def __init__(self, base: Retriever, rr: Reranker,
                     all_chunks: list, fetch_k: int = 10) -> None:
            self._base = base
            self._rr = rr
            self._fetch_k = fetch_k
            self.store = type("S", (), {"chunks": all_chunks})()

        def retrieve(self, query: str, k: int, min_score: float) -> list:
            candidates = self._base.retrieve(
                query, k=max(self._fetch_k, k * 2), min_score=min_score
            )
            return self._rr.rerank(query, candidates, top_n=k)

    print("\nLoading cross-encoder re-ranker…", flush=True)
    reranker = Reranker()
    print(f"  Model : {reranker.model_name}")
    reranker_adapter = _RerankerAdapter(retriever, reranker, chunks)

    # ── Dense strategy comparison (fixed-size vs sentence-boundary) ──────────
    for k in (1, 3, 5):
        print(section(f"Dense Retrieval Quality  k={k}"))
        comparison = compare_chunking_strategies(corpus, embedder, SAMPLES, k=k)
        print(metrics_table(comparison, k=k))

    # ── Dense / Hybrid / Reranked comparison (k=1/3/5) ──────────────────────
    for k in (1, 3, 5):
        print(section(f"Dense vs Hybrid vs Reranked  k={k}"))
        dense_metrics = evaluate_retrieval(retriever, SAMPLES, k=k)
        hybrid_metrics = evaluate_retrieval(hybrid_adapter, SAMPLES, k=k)
        reranked_metrics = evaluate_retrieval(reranker_adapter, SAMPLES, k=k)
        print(metrics_table({
            "dense": dense_metrics,
            "hybrid (BM25+dense)": hybrid_metrics,
            "dense + reranker": reranked_metrics,
        }, k=k))

    # ── Per-query breakdown (k=1, fixed-size) ────────────────────────────────
    print(section("Per-Query Breakdown  (fixed-size, k=1)"))
    print(f"  {'Query':<40} {'Dense':>8}  {'Hybrid':>8}  {'Reranked':>10}  Match?")
    print("  " + "-" * (W - 2))
    for sample in SAMPLES:
        d_results = retriever.retrieve(sample.query, k=1)
        h_results = hybrid.retrieve(sample.query, k=1)
        r_results = reranker_adapter.retrieve(sample.query, k=1, min_score=0.0)
        top_chunk, d_score = d_results[0]
        _, h_score = h_results[0]
        _, r_score = r_results[0]
        hit = "✓" if any(kw.lower() in top_chunk.text.lower()
                         for kw in sample.relevant_keywords) else "✗"
        truncated = sample.query[:38] + ".." if len(sample.query) > 40 else sample.query
        print(f"  {truncated:<40} {d_score:>8.4f}  {h_score:>8.4f}  {r_score:>10.2f}  {hit}")

    # ── Ingestion timing ─────────────────────────────────────────────────────
    print(section("Ingestion Timing"))
    t_fixed = time_ingest(corpus, embedder, chunk_size=512, overlap=64)

    from src.chunker import chunk_by_sentence
    import time
    t0 = time.perf_counter()
    s_chunks = chunk_by_sentence(corpus, max_chunk_size=512)
    t1 = time.perf_counter()
    s_embs = embedder.embed([c.text for c in s_chunks])
    t2 = time.perf_counter()
    s_store = VectorStore(embedder.dimension)
    s_store.add(s_chunks, s_embs)
    t3 = time.perf_counter()

    from src.benchmark import IngestTiming
    t_sentence = IngestTiming(
        chunk_ms=(t1 - t0) * 1000,
        embed_ms=(t2 - t1) * 1000,
        index_ms=(t3 - t2) * 1000,
        total_ms=(t3 - t0) * 1000,
        num_chunks=len(s_chunks),
    )

    print(timing_block("Fixed-size  (chunk_size=512, overlap=64)", t_fixed))
    print()
    print(timing_block("Sentence-boundary  (max_chunk_size=512)", t_sentence))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(" Done.")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()
