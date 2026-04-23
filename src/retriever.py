"""
Retrieval module — bridges the Embedder and VectorStore to turn a plain text
query into a ranked list of relevant chunks.
"""

from src.chunker import Chunk
from src.embedder import Embedder
from src.vector_store import VectorStore


class Retriever:
    """
    Converts a query string into a ranked list of relevant Chunk objects.

    This is the glue layer between user input and the vector store.
    It embeds the query with the same model used to embed the documents,
    then delegates to VectorStore.search().

    Args:
        embedder: An initialised Embedder instance.
        store: A populated VectorStore instance.
    """

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """
        Retrieve the top-k chunks most relevant to a query.

        Args:
            query: The user's question or search string.
            k: Maximum number of chunks to return.
            min_score: Cosine similarity threshold. Chunks with a score
                       below this value are dropped from the results.
                       Use 0.0 (default) to return all k results regardless
                       of score. A value around 0.3–0.4 filters out weak
                       matches in practice.

        Returns:
            List of (Chunk, score) tuples sorted by descending similarity,
            filtered to scores >= min_score.

        Raises:
            ValueError: If query is empty, or if the store is empty.
        """
        if not query.strip():
            raise ValueError("query must not be empty")

        query_embedding = self.embedder.embed_one(query)
        results = self.store.search(query_embedding, k=k)

        if min_score > 0.0:
            results = [(chunk, score) for chunk, score in results if score >= min_score]

        return results
