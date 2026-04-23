"""
Document store module — tracks all ingested documents so the pipeline
can support multiple PDFs and remove individual ones.

The core problem with multi-document RAG is that FAISS does not support
removing individual vectors. To delete a document, you must rebuild the
index from scratch using the remaining documents' chunks and embeddings.

DocumentStore solves this by keeping each document's chunks and embeddings
in memory alongside the FAISS index. When a document is added or removed,
DocumentStore rebuilds the VectorStore from the retained records. For
portfolio-scale corpora (tens of PDFs, thousands of chunks) this is fast
enough that rebuilding on every change is fine.

Typical usage
─────────────
::

    store = DocumentStore(dimension=384)

    # Add documents
    store.add("paper.pdf", chunks, embeddings)
    store.add("notes.pdf", other_chunks, other_embeddings)

    # Build the combined search index
    vector_store = store.build_vector_store()
    retriever    = Retriever(embedder, vector_store)

    # List what's in the store
    for rec in store.list_documents():
        print(rec.source, rec.chunk_count)

    # Remove a document and rebuild
    store.remove("paper.pdf")
    vector_store = store.build_vector_store()
    retriever    = Retriever(embedder, vector_store)
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from src.chunker import Chunk
from src.vector_store import VectorStore


@dataclass
class DocumentRecord:
    """
    Metadata and content for a single ingested document.

    Stored in full so the combined VectorStore can be rebuilt after
    any document is added or removed.

    Attributes:
        source:      Document identifier, typically the original filename.
        chunk_count: Number of chunks extracted from this document.
        ingested_at: UTC timestamp of when the document was ingested.
        chunks:      Chunk objects produced by the chunker.
        embeddings:  Float32 array of shape (chunk_count, dimension).
    """

    source: str
    chunk_count: int
    ingested_at: datetime
    chunks: list[Chunk]
    embeddings: np.ndarray


class DocumentStore:
    """
    Registry of all ingested documents with support for add and remove.

    Maintains a dict of DocumentRecord objects keyed by source name.
    Python 3.7+ dicts preserve insertion order, so documents are listed
    in ingestion order throughout.

    When a document is added with a source name that already exists, the
    old record is replaced — this is the natural behaviour for "re-ingest
    the same file after editing it."

    Args:
        dimension: Embedding vector length. Must match the Embedder used
                   to produce the vectors (384 for all-MiniLM-L6-v2).
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._records: dict[str, DocumentRecord] = {}

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def add(
        self,
        source: str,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add or replace a document in the store.

        If a record with the same source name already exists it is
        overwritten — safe to re-ingest the same filename after changes.

        Args:
            source:     Document identifier (typically the filename).
            chunks:     Chunk objects from the chunker.
            embeddings: Float32 array of shape (len(chunks), dimension).

        Raises:
            ValueError: If source is empty or chunks/embeddings lengths differ.
        """
        if not source.strip():
            raise ValueError("source must not be empty")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        self._records[source] = DocumentRecord(
            source=source,
            chunk_count=len(chunks),
            ingested_at=datetime.now(UTC),
            chunks=chunks,
            embeddings=np.array(embeddings, dtype=np.float32),
        )

    def remove(self, source: str) -> bool:
        """
        Remove a document by source name.

        Args:
            source: Document identifier to remove.

        Returns:
            True if the document was found and removed, False otherwise.
        """
        if source in self._records:
            del self._records[source]
            return True
        return False

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_vector_store(self) -> VectorStore:
        """
        Construct a fresh VectorStore from all stored documents.

        Iterates over records in insertion order and adds each document's
        chunks and embeddings. The resulting VectorStore is independent of
        this DocumentStore — mutating one does not affect the other.

        Returns:
            A populated VectorStore ready for similarity search.

        Raises:
            ValueError: If no documents have been ingested yet.
        """
        if not self._records:
            raise ValueError(
                "DocumentStore is empty — ingest at least one document first"
            )

        store = VectorStore(self.dimension)
        for record in self._records.values():
            store.add(record.chunks, record.embeddings)
        return store

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def all_chunks(self) -> list[Chunk]:
        """
        Return all chunks across all documents, in ingestion order.

        Useful for building a BM25Store over the full corpus.
        """
        chunks: list[Chunk] = []
        for record in self._records.values():
            chunks.extend(record.chunks)
        return chunks

    def list_documents(self) -> list[DocumentRecord]:
        """Return all document records in ingestion order."""
        return list(self._records.values())

    @property
    def document_count(self) -> int:
        """Number of distinct documents currently in the store."""
        return len(self._records)

    @property
    def total_chunks(self) -> int:
        """Total number of chunks across all stored documents."""
        return sum(r.chunk_count for r in self._records.values())

    def __len__(self) -> int:
        """Return the number of documents (not chunks) in the store."""
        return len(self._records)

    def __contains__(self, source: str) -> bool:
        """Return True if a document with the given source name exists."""
        return source in self._records
