"""
Vector store module — a FAISS-backed index that stores chunk embeddings and
supports similarity search, persistence, and metadata retrieval.
"""

import pickle
from pathlib import Path

import faiss
import numpy as np

from src.chunker import Chunk


class VectorStore:
    """
    Thin wrapper around a FAISS flat inner-product index.

    Because embeddings from Embedder are L2-normalised, dot product equals
    cosine similarity. Search results therefore return scores in [-1, 1]
    where higher means more semantically similar.

    The chunks list acts as a parallel array to the FAISS index: the i-th
    entry in self.chunks corresponds to the i-th vector in the index.

    Args:
        dimension: Length of each embedding vector. Must match the Embedder
                   used to produce the vectors (384 for all-MiniLM-L6-v2).
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        # IndexFlatIP: exact (no approximation) inner-product search.
        # Fine for thousands to low-millions of vectors on CPU.
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their corresponding embeddings to the store.

        Args:
            chunks: List of Chunk objects from the chunker.
            embeddings: 2-D float32 array of shape (len(chunks), dimension).

        Raises:
            ValueError: If the lengths of chunks and embeddings do not match,
                        or if the embedding dimension is wrong.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.dimension}), "
                f"got {embeddings.shape}"
            )

        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """
        Return the k most similar chunks to a query embedding.

        Args:
            query_embedding: 1-D or 2-D float32 array. If 1-D it is reshaped
                             automatically. Must be L2-normalised.
            k: Number of results to return. Capped at the index size.

        Returns:
            List of (Chunk, score) tuples, sorted by descending similarity.
            Scores are cosine similarities in [-1, 1].

        Raises:
            ValueError: If the store is empty.
        """
        if self.index.ntotal == 0:
            raise ValueError("Vector store is empty — add chunks before searching")

        k = min(k, self.index.ntotal)

        query = np.array(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        scores, indices = self.index.search(query, k)

        results: list[tuple[Chunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:  # FAISS returns -1 for padding when k > ntotal
                results.append((self.chunks[idx], float(score)))

        return results

    # ------------------------------------------------------------------
    # Size helper
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of vectors currently in the index."""
        return self.index.ntotal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """
        Persist the FAISS index and chunk metadata to a directory.

        Creates two files:
          {directory}/index.faiss   — the FAISS binary index
          {directory}/chunks.pkl    — the list of Chunk objects

        Args:
            directory: Path to the save directory (created if absent).
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))

        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """
        Load a previously saved VectorStore from a directory.

        Args:
            directory: Path to the directory produced by save().

        Returns:
            A fully restored VectorStore instance.

        Raises:
            FileNotFoundError: If either expected file is missing.
        """
        directory = Path(directory)
        index_path = directory / "index.faiss"
        chunks_path = directory / "chunks.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        index = faiss.read_index(str(index_path))

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        store = cls(dimension=index.d)
        store.index = index
        store.chunks = chunks
        return store
