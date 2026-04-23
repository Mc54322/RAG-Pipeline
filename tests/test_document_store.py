"""
Tests for multi-document storage (src/document_store.py).

Covers DocumentStore: add, remove, build_vector_store, all_chunks,
list_documents, and container protocol (__len__, __contains__).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

import numpy as np
import pytest

from src.chunker import Chunk
from src.document_store import DocumentRecord, DocumentStore
from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIMENSION = 4


def _rand_embeddings(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, DIMENSION)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _make_chunks(n: int, source: str = "test.pdf") -> list[Chunk]:
    return [
        Chunk(text=f"Chunk {i} from {source}", index=i,
              start_char=i * 20, end_char=i * 20 + 19,
              metadata={"source": source})
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestDocumentStoreInit:
    def test_starts_empty(self):
        assert len(DocumentStore(DIMENSION)) == 0

    def test_document_count_zero(self):
        assert DocumentStore(DIMENSION).document_count == 0

    def test_total_chunks_zero(self):
        assert DocumentStore(DIMENSION).total_chunks == 0

    def test_list_documents_empty(self):
        assert DocumentStore(DIMENSION).list_documents() == []

    def test_all_chunks_empty(self):
        assert DocumentStore(DIMENSION).all_chunks() == []

    def test_not_contains_before_add(self):
        assert "paper.pdf" not in DocumentStore(DIMENSION)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

class TestDocumentStoreAdd:
    def test_add_increments_document_count(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(3), _rand_embeddings(3))
        assert ds.document_count == 1

    def test_add_increments_total_chunks(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(3), _rand_embeddings(3))
        assert ds.total_chunks == 3

    def test_two_adds_accumulate(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(3), _rand_embeddings(3, seed=0))
        ds.add("b.pdf", _make_chunks(5), _rand_embeddings(5, seed=1))
        assert ds.document_count == 2
        assert ds.total_chunks == 8

    def test_add_replaces_existing_source(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(3), _rand_embeddings(3))
        ds.add("a.pdf", _make_chunks(7), _rand_embeddings(7))
        assert ds.document_count == 1
        assert ds.total_chunks == 7

    def test_contains_after_add(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(2), _rand_embeddings(2))
        assert "a.pdf" in ds

    def test_add_empty_source_raises(self):
        ds = DocumentStore(DIMENSION)
        with pytest.raises(ValueError):
            ds.add("   ", _make_chunks(2), _rand_embeddings(2))

    def test_add_mismatched_lengths_raises(self):
        ds = DocumentStore(DIMENSION)
        with pytest.raises(ValueError):
            ds.add("a.pdf", _make_chunks(3), _rand_embeddings(5))

    def test_record_chunk_count_correct(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(4), _rand_embeddings(4))
        assert ds.list_documents()[0].chunk_count == 4

    def test_record_source_correct(self):
        ds = DocumentStore(DIMENSION)
        ds.add("my_paper.pdf", _make_chunks(2), _rand_embeddings(2))
        assert ds.list_documents()[0].source == "my_paper.pdf"

    def test_record_ingested_at_is_set(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(2), _rand_embeddings(2))
        assert isinstance(ds.list_documents()[0].ingested_at, datetime)


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

class TestDocumentStoreRemove:
    def test_remove_returns_true_when_found(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(2), _rand_embeddings(2))
        assert ds.remove("a.pdf") is True

    def test_remove_returns_false_when_missing(self):
        assert DocumentStore(DIMENSION).remove("nonexistent.pdf") is False

    def test_remove_decrements_count(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(2), _rand_embeddings(2))
        ds.add("b.pdf", _make_chunks(3), _rand_embeddings(3))
        ds.remove("a.pdf")
        assert ds.document_count == 1

    def test_remove_reduces_total_chunks(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(2), _rand_embeddings(2))
        ds.add("b.pdf", _make_chunks(3), _rand_embeddings(3))
        ds.remove("a.pdf")
        assert ds.total_chunks == 3

    def test_remove_not_contains_after(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(2), _rand_embeddings(2))
        ds.remove("a.pdf")
        assert "a.pdf" not in ds

    def test_all_chunks_excludes_removed_document(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(2, source="a.pdf"), _rand_embeddings(2, seed=0))
        ds.add("b.pdf", _make_chunks(3, source="b.pdf"), _rand_embeddings(3, seed=1))
        ds.remove("a.pdf")
        sources = {c.metadata["source"] for c in ds.all_chunks()}
        assert "a.pdf" not in sources
        assert "b.pdf" in sources

    def test_remove_last_document_leaves_empty(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(2), _rand_embeddings(2))
        ds.remove("a.pdf")
        assert ds.document_count == 0
        assert ds.total_chunks == 0


# ---------------------------------------------------------------------------
# all_chunks
# ---------------------------------------------------------------------------

class TestDocumentStoreAllChunks:
    def test_concatenates_all_chunks(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(2, "a.pdf"), _rand_embeddings(2, seed=0))
        ds.add("b.pdf", _make_chunks(3, "b.pdf"), _rand_embeddings(3, seed=1))
        all_chunks = ds.all_chunks()
        assert len(all_chunks) == 5
        assert all_chunks[0].metadata["source"] == "a.pdf"
        assert all_chunks[2].metadata["source"] == "b.pdf"


# ---------------------------------------------------------------------------
# build_vector_store
# ---------------------------------------------------------------------------

class TestDocumentStoreBuildVectorStore:
    def test_build_raises_when_empty(self):
        with pytest.raises(ValueError):
            DocumentStore(DIMENSION).build_vector_store()

    def test_build_returns_vector_store(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(3), _rand_embeddings(3))
        assert isinstance(ds.build_vector_store(), VectorStore)

    def test_build_has_correct_chunk_count(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(3), _rand_embeddings(3))
        assert len(ds.build_vector_store()) == 3

    def test_build_two_documents_total_chunks(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(3), _rand_embeddings(3, seed=0))
        ds.add("b.pdf", _make_chunks(5), _rand_embeddings(5, seed=1))
        assert len(ds.build_vector_store()) == 8

    def test_build_after_remove_has_fewer_chunks(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(3), _rand_embeddings(3, seed=0))
        ds.add("b.pdf", _make_chunks(5), _rand_embeddings(5, seed=1))
        ds.remove("a.pdf")
        assert len(ds.build_vector_store()) == 5

    def test_build_is_searchable(self):
        ds = DocumentStore(DIMENSION)
        ds.add("a.pdf", _make_chunks(3), _rand_embeddings(3))
        vs = ds.build_vector_store()
        results = vs.search(_rand_embeddings(1)[0], k=1)
        assert len(results) == 1
