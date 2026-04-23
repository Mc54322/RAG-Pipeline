"""
PDF ingestion module — extracts and cleans text from PDF files using PyMuPDF.
"""

import re
from pathlib import Path

import fitz  # PyMuPDF


def load_pdf(path: str | Path) -> str:
    """
    Extract all text from a PDF file.

    Args:
        path: Path to the PDF file.

    Returns:
        Concatenated text content of all pages.

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
        ValueError: If the file is not a PDF.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    pages: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            pages.append(page.get_text())

    return "\n".join(pages)


def clean_text(text: str) -> str:
    """
    Normalize extracted PDF text by collapsing whitespace and removing
    common extraction artefacts.

    Args:
        text: Raw text extracted from a PDF.

    Returns:
        Cleaned text string.
    """
    # Collapse runs of whitespace/newlines into a single space
    text = re.sub(r"\s+", " ", text)
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()
