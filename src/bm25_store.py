"""
BM25 index module — keyword-based retrieval over a list of Chunks.

BM25 (Best Match 25) is a probabilistic ranking function widely used in
search engines. Unlike dense embeddings, BM25 operates on exact token
matches with term-frequency and inverse-document-frequency weighting. It
excels at queries that contain rare, specific keywords (product codes,
names, technical terms) that a semantic model may not handle well.

This module wraps rank_bm25.BM25Okapi, which implements the standard
Okapi BM25 variant (k1=1.5, b=0.75 defaults).
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.chunker import Chunk


class BM25Store:
    """
    BM25 keyword index over a fixed list of Chunks.

    Constructed from a list of Chunk objects. The index is built once at
    construction time; adding new chunks requires creating a new BM25Store.
    This mirrors how BM25Okapi works — it precomputes IDF statistics over
    the full corpus and cannot be updated incrementally.

    Args:
        chunks: The document chunks to index. Order is preserved and used
                to map search results back to Chunk objects.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25Store requires at least one chunk")
        self.chunks = chunks
        tokenized = [self.tokenize(c.text) for c in chunks]
        self._index = BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Convert text to a list of lowercase tokens for BM25 indexing.

        Splits on any non-alphanumeric character and drops single-character
        tokens (mostly punctuation artefacts). Lowercasing ensures that
        "Python" and "python" match.

        Args:
            text: Raw or cleaned text to tokenise.

        Returns:
            List of lowercase string tokens, each at least 2 characters long.
        """
        tokens = re.split(r"[^a-z0-9]+", text.lower())
        return [t for t in tokens if len(t) > 1]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """
        Return the top-k chunks ranked by BM25 score for the given query.

        BM25 scores are unbounded non-negative floats: higher means more
        keyword-relevant. A score of 0.0 means no query terms appeared in
        the chunk. Unlike cosine similarity, scores are not normalised to
        a fixed range — they depend on corpus size and term frequencies.

        Args:
            query: The user's question or search string.
            k: Maximum number of results to return.

        Returns:
            List of (Chunk, bm25_score) tuples sorted by descending score.
            Returns an empty list if the query tokenises to nothing (e.g.
            a query consisting entirely of punctuation).

        Raises:
            ValueError: If the store is empty (no chunks were indexed).
        """
        if not self.chunks:
            raise ValueError("BM25Store is empty — add chunks before searching")

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores: np.ndarray = self._index.get_scores(tokens)

        # Pair each chunk with its score and sort descending
        pairs = sorted(
            zip(self.chunks, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(chunk, score) for chunk, score in pairs[:k]]

    # ------------------------------------------------------------------
    # Size helper
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of chunks in the index."""
        return len(self.chunks)
