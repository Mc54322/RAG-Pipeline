"""
Tests for PDF ingestion and text cleaning (src/ingestion.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import fitz  # PyMuPDF

from src.ingestion import clean_text, load_pdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pdf(tmp_path: Path, text: str) -> Path:
    """Create a single-page PDF containing *text* and return its path."""
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# load_pdf
# ---------------------------------------------------------------------------

class TestLoadPdf:
    def test_returns_string(self, tmp_path):
        pdf = make_pdf(tmp_path, "Hello RAG world")
        assert isinstance(load_pdf(pdf), str)

    def test_result_non_empty(self, tmp_path):
        pdf = make_pdf(tmp_path, "Some content")
        assert len(load_pdf(pdf)) > 0

    def test_contains_expected_text(self, tmp_path):
        pdf = make_pdf(tmp_path, "Hello RAG world")
        assert "Hello RAG world" in load_pdf(pdf)

    def test_accepts_path_object(self, tmp_path):
        pdf = make_pdf(tmp_path, "path object test")
        load_pdf(Path(pdf))  # should not raise

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pdf(tmp_path / "nonexistent.pdf")

    def test_non_pdf_raises(self, tmp_path):
        txt = tmp_path / "file.txt"
        txt.write_text("hello")
        with pytest.raises(ValueError):
            load_pdf(txt)

    def test_multipage_pdf_concatenated(self, tmp_path):
        """Text from all pages should appear in the result."""
        pdf_path = tmp_path / "multi.pdf"
        doc = fitz.open()
        for text in ["Page one content.", "Page two content."]:
            page = doc.new_page()
            page.insert_text((72, 72), text)
        doc.save(pdf_path)
        doc.close()
        result = load_pdf(pdf_path)
        assert "Page one content." in result
        assert "Page two content." in result


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_collapses_whitespace(self):
        assert clean_text("hello   world") == "hello world"

    def test_collapses_newlines(self):
        assert clean_text("line1\n\nline2") == "line1 line2"

    def test_strips_leading_trailing(self):
        assert clean_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_preserves_content(self):
        result = clean_text("  The  quick   brown  fox  ")
        assert "quick" in result
        assert "fox" in result

    def test_mixed_whitespace(self):
        assert clean_text("a\t\n b") == "a b"
