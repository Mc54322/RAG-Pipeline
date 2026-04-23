"""
Text chunking module — splits cleaned text into overlapping chunks
suitable for embedding and retrieval.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single text chunk with its position metadata."""

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    source: str = "",
) -> list[Chunk]:
    """
    Split text into overlapping fixed-size chunks (by character count).

    Overlap ensures context is preserved across chunk boundaries, which
    improves retrieval quality for sentences that straddle a split point.

    Args:
        text: Cleaned input text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters to repeat between adjacent chunks.
        source: Optional source label (e.g. filename) stored in metadata.

    Returns:
        List of Chunk objects in document order.

    Raises:
        ValueError: If chunk_size <= 0 or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_slice = text[start:end]

        chunks.append(
            Chunk(
                text=chunk_text_slice,
                index=index,
                start_char=start,
                end_char=end,
                metadata={"source": source},
            )
        )

        if end == len(text):
            break

        start += step
        index += 1

    return chunks


def chunk_by_sentence(
    text: str,
    max_chunk_size: int = 512,
    overlap_sentences: int = 1,
    source: str = "",
) -> list[Chunk]:
    """
    Split text into chunks at sentence boundaries to avoid mid-sentence cuts.

    Sentences are accumulated until adding the next would exceed
    max_chunk_size, at which point a new chunk is started. The last
    overlap_sentences of the previous chunk are prepended to maintain
    context continuity.

    Args:
        text: Cleaned input text to split.
        max_chunk_size: Soft character limit per chunk.
        overlap_sentences: Number of sentences to carry over between chunks.
        source: Optional source label stored in metadata.

    Returns:
        List of Chunk objects in document order.
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_len = 0
    char_cursor = 0
    index = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_len + sentence_len > max_chunk_size and current_sentences:
            chunk_text_str = " ".join(current_sentences)
            start_char = char_cursor - current_len
            chunks.append(
                Chunk(
                    text=chunk_text_str,
                    index=index,
                    start_char=max(0, start_char),
                    end_char=char_cursor,
                    metadata={"source": source},
                )
            )
            index += 1
            # Carry over overlap sentences
            current_sentences = current_sentences[-overlap_sentences:]
            current_len = sum(len(s) for s in current_sentences) + len(current_sentences)

        current_sentences.append(sentence)
        current_len += sentence_len + 1  # +1 for space
        char_cursor += sentence_len + 1

    # Flush any remaining sentences
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        start_char = char_cursor - current_len
        chunks.append(
            Chunk(
                text=chunk_text_str,
                index=index,
                start_char=max(0, start_char),
                end_char=char_cursor,
                metadata={"source": source},
            )
        )

    return chunks
