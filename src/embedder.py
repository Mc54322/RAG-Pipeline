"""
Embedding module — converts text into dense vector representations using
a local sentence-transformers model. No API calls, no cost, runs on CPU.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wraps a sentence-transformers model to produce L2-normalised embeddings.

    Normalisation means every vector has length 1, which turns a dot-product
    similarity (IndexFlatIP in FAISS) into cosine similarity — a value in
    [-1, 1] where 1 is identical meaning and -1 is opposite.

    Args:
        model_name: HuggingFace model ID. Defaults to all-MiniLM-L6-v2,
                    a 22M-parameter model that produces 384-dim embeddings.
                    It is small enough to run comfortably on CPU and strong
                    enough for a solid retrieval baseline.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()

    def embed(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of texts into a 2-D float32 array.

        Args:
            texts: Strings to embed. Order is preserved.
            batch_size: Number of texts processed per forward pass.
                        Larger batches are faster but use more RAM.
            show_progress: Show a tqdm progress bar (useful for large corpora).

        Returns:
            Array of shape (len(texts), self.dimension), dtype float32,
            with each row L2-normalised to unit length.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("texts list must not be empty")

        embeddings: np.ndarray = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalise so dot product == cosine sim
        )
        return embeddings.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """
        Embed a single string. Convenience wrapper around embed().

        Args:
            text: The string to embed (typically a user query).

        Returns:
            1-D float32 array of length self.dimension, L2-normalised.
        """
        return self.embed([text])[0]
