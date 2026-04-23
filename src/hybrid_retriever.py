"""
Hybrid retrieval module — fuses BM25 keyword search with dense vector
search using Reciprocal Rank Fusion (RRF).

Why hybrid?
  Dense retrieval (embeddings) captures semantic similarity but can miss
  exact keyword matches, especially for rare terms, proper nouns, and
  technical identifiers. BM25 is the opposite: excellent at keyword
  matching but blind to synonyms and paraphrases.

  Combining both via RRF reliably outperforms either alone across a wide
  variety of query types — this is the standard approach in production RAG
  systems (e.g. Elasticsearch, Weaviate, Pinecone hybrid search).

Reciprocal Rank Fusion (RRF):
  For each chunk, its RRF score is the sum of reciprocal ranks from each
  retriever:

      rrf_score = Σ  1 / (rrf_k + rank_i)

  where rank_i is the 1-indexed position of the chunk in retriever i's
  result list, and rrf_k (default 60) is a smoothing constant from the
  original paper (Cormack et al., 2009). Chunks not returned by a
  retriever contribute 0 from that retriever.

  RRF advantages over weighted score fusion:
    - No score normalisation required (BM25 and cosine scores have
      different ranges and distributions)
    - No alpha hyperparameter to tune
    - Empirically robust across domains
"""

from src.bm25_store import BM25Store
from src.chunker import Chunk
from src.retriever import Retriever


class HybridRetriever:
    """
    Combines dense and BM25 retrieval via Reciprocal Rank Fusion.

    Both retrievers must be built over the same set of chunks (same Python
    objects), so that chunk identity can be used to merge their ranked
    lists during fusion.

    Args:
        retriever: A populated dense Retriever (Embedder + VectorStore).
        bm25_store: A BM25Store built from the same chunk list.
        rrf_k: RRF smoothing constant. The original paper recommends 60;
               lower values amplify the difference between rank 1 and 2,
               higher values flatten it.
    """

    def __init__(
        self,
        retriever: Retriever,
        bm25_store: BM25Store,
        rrf_k: int = 60,
    ) -> None:
        self.retriever = retriever
        self.bm25_store = bm25_store
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """
        Retrieve the top-k chunks using hybrid BM25 + dense fusion.

        Steps:
          1. Fetch up to 2*k results from the dense retriever
          2. Fetch up to 2*k results from BM25
          3. Fuse ranked lists with RRF
          4. Apply min_score threshold (on RRF score, not cosine)
          5. Return top-k

        Fetching 2*k from each retriever before fusing gives the fusion
        step enough candidates to work with — a chunk ranked 6th by dense
        but 1st by BM25 should surface in the final top-k.

        Args:
            query: The user's question or search string.
            k: Maximum number of chunks to return.
            min_score: Minimum RRF score threshold. RRF scores are small
                       positive floats (typically 0.005–0.03 for two
                       retrievers with rrf_k=60). Use 0.0 to disable.

        Returns:
            List of (Chunk, rrf_score) tuples sorted by descending score.

        Raises:
            ValueError: If query is empty.
        """
        if not query.strip():
            raise ValueError("query must not be empty")

        fetch_k = max(k * 2, 10)

        dense_results = self.retriever.retrieve(query, k=fetch_k, min_score=0.0)
        bm25_results = self.bm25_store.search(query, k=fetch_k)

        fused = self._rrf_fuse(dense_results, bm25_results)

        if min_score > 0.0:
            fused = [(chunk, score) for chunk, score in fused if score >= min_score]

        return fused[:k]

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        dense_results: list[tuple[Chunk, float]],
        bm25_results: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """
        Apply Reciprocal Rank Fusion to two ranked result lists.

        Uses Python's object identity (id()) to recognise the same Chunk
        across both lists. This is valid because both retrievers are built
        from the same chunk list — the BM25Store holds a reference to the
        same Chunk objects as the VectorStore.

        Args:
            dense_results: Ranked (Chunk, score) list from the dense retriever.
            bm25_results: Ranked (Chunk, score) list from BM25.

        Returns:
            Fused list of (Chunk, rrf_score) sorted by descending score.
        """
        rrf_scores: dict[int, float] = {}   # chunk id → accumulated RRF score
        chunk_map: dict[int, Chunk] = {}    # chunk id → Chunk object

        for rank, (chunk, _) in enumerate(dense_results):
            cid = id(chunk)
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, (chunk, _) in enumerate(bm25_results):
            cid = id(chunk)
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        return [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_ids]
