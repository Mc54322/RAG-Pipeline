"""
Cross-encoder re-ranking module — improves retrieval precision by scoring
each (query, passage) pair jointly.

Two-stage retrieval pipeline
─────────────────────────────
Stage 1 — recall: bi-encoder (dense) or hybrid retrieval, fast, fetch k=20–50
Stage 2 — precision: cross-encoder re-ranking, slow, return top-n (n << k)

Why two stages?
  A bi-encoder embeds the query and each document independently, then
  compares their vectors with a dot product. This is fast because document
  embeddings are pre-computed, but it loses the fine-grained interaction
  between query and document tokens.

  A cross-encoder processes (query, document) as a single concatenated
  input: [CLS] query [SEP] document [SEP]. Every query token can attend to
  every document token through the self-attention layers, giving a much
  more accurate relevance judgment. The trade-off is that you cannot
  pre-compute anything — inference must run for every candidate at query
  time.

  The two-stage approach gets the best of both: bi-encoder narrows the
  search space cheaply (e.g. 100,000 chunks → 20 candidates), then the
  cross-encoder accurately ranks those 20 candidates.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  Trained on MS MARCO, a large-scale passage retrieval dataset of ~8.8M
  passages with human-labelled relevance judgements. The MiniLM-L6 variant
  has 6 transformer layers and ~22M parameters — the same size footprint as
  the bi-encoder, running comfortably on CPU.

  Output is a single raw logit (not a probability). Higher = more relevant.
  Scores can be negative. Do not interpret them as probabilities or compare
  them across different models.
"""

import numpy as np
from sentence_transformers import CrossEncoder

from src.chunker import Chunk


class Reranker:
    """
    Cross-encoder re-ranker for a list of retrieved candidates.

    Scores each (query, passage) pair jointly using a cross-encoder model,
    then re-orders the candidates by descending score. Typically used after
    a first-stage retriever to improve precision within the top results.

    Args:
        model_name: HuggingFace model identifier. Defaults to
            cross-encoder/ms-marco-MiniLM-L-6-v2, a strong passage
            re-ranking model trained on MS MARCO.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_n: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Re-rank a list of retrieved (Chunk, score) pairs using the cross-encoder.

        The incoming first-stage scores (cosine, BM25, or RRF) are discarded
        and replaced with cross-encoder logits. The cross-encoder's judgment
        is treated as authoritative over the first-stage score.

        Args:
            query: The user's question or search string.
            candidates: Ordered list of (Chunk, first_stage_score) from the
                        first-stage retriever. Order does not matter — all
                        candidates are scored and re-sorted.
            top_n: If provided, return only the top_n highest-scoring chunks.
                   None returns all candidates re-ranked.

        Returns:
            List of (Chunk, cross_encoder_score) tuples sorted by descending
            score. Cross-encoder scores are raw logits (unbounded floats);
            higher means more relevant.

        Raises:
            ValueError: If query is empty.
        """
        if not query.strip():
            raise ValueError("query must not be empty")

        if not candidates:
            return []

        pairs = [(query, chunk.text) for chunk, _ in candidates]
        raw_scores: np.ndarray = self._model.predict(pairs)

        reranked = sorted(
            zip([chunk for chunk, _ in candidates], raw_scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        if top_n is not None:
            reranked = reranked[:top_n]

        return [(chunk, float(score)) for chunk, score in reranked]
