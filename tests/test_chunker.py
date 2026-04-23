"""
Tests for text chunking strategies (src/chunker.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.chunker import Chunk, chunk_by_sentence, chunk_text


SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly."
)


# ---------------------------------------------------------------------------
# chunk_text — fixed-size overlapping chunks
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_returns_list_of_chunks(self):
        chunks = chunk_text(SAMPLE_TEXT)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_single_chunk_for_short_text(self):
        chunks = chunk_text("short", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].text == "short"

    def test_multiple_chunks_for_long_text(self):
        chunks = chunk_text("x" * 1000, chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_overlap_content(self):
        text = "a" * 200
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # The end of chunk 0 should equal the start of chunk 1
        assert chunks[0].text[-20:] == chunks[1].text[:20]

    def test_full_coverage(self):
        """Every character position in the original text appears in at least one chunk."""
        text = "abcdefghij" * 10
        chunks = chunk_text(text, chunk_size=15, overlap=5)
        covered = set()
        for c in chunks:
            for i in range(len(c.text)):
                covered.add(c.start_char + i)
        assert covered == set(range(len(text)))

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size=0)

    def test_overlap_gte_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size=10, overlap=10)

    def test_metadata_source(self):
        chunks = chunk_text("hello world", source="doc.pdf")
        assert chunks[0].metadata["source"] == "doc.pdf"

    def test_indices_are_sequential(self):
        chunks = chunk_text("x" * 500, chunk_size=100, overlap=10)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_start_end_chars_consistent(self):
        text = "hello world"
        chunks = chunk_text(text, chunk_size=6, overlap=0)
        for c in chunks:
            assert text[c.start_char:c.end_char] == c.text

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []


# ---------------------------------------------------------------------------
# chunk_by_sentence — sentence-boundary chunking
# ---------------------------------------------------------------------------

class TestChunkBySentence:
    def test_returns_list_of_chunks(self):
        chunks = chunk_by_sentence(SAMPLE_TEXT)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_no_mid_sentence_cuts(self):
        """Each chunk should end with sentence-terminating punctuation."""
        chunks = chunk_by_sentence(SAMPLE_TEXT, max_chunk_size=80)
        for c in chunks:
            assert c.text.rstrip().endswith((".", "!", "?"))

    def test_single_chunk_for_short_text(self):
        chunks = chunk_by_sentence("One sentence only.", max_chunk_size=512)
        assert len(chunks) == 1

    def test_source_metadata(self):
        chunks = chunk_by_sentence(SAMPLE_TEXT, source="paper.pdf")
        for c in chunks:
            assert c.metadata["source"] == "paper.pdf"

    def test_all_sentences_present(self):
        """All sentences from the original text should appear in some chunk."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", SAMPLE_TEXT.strip())
        chunks = chunk_by_sentence(SAMPLE_TEXT, max_chunk_size=80)
        combined = " ".join(c.text for c in chunks)
        for sentence in sentences:
            assert sentence in combined

    def test_multi_sentence_accumulation(self):
        """Short sentences should be accumulated into one chunk."""
        text = "A. B. C. D. E."
        chunks = chunk_by_sentence(text, max_chunk_size=20)
        # Multiple sentences should fit in one chunk
        assert any(c.text.count(".") > 1 for c in chunks)
