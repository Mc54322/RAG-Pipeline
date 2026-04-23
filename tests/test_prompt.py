"""
Tests for prompt construction (src/prompt.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.chunker import Chunk
from src.prompt import SYSTEM_PROMPT, build_prompt, get_system_prompt


def make_chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(text=text, index=index, start_char=0, end_char=len(text),
                 metadata={"source": "test"})


def make_chunks(texts: list[str]) -> list[tuple[Chunk, float]]:
    return [(make_chunk(t, i), round(0.9 - i * 0.1, 2)) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_returns_string(self):
        result = build_prompt("What is AI?", make_chunks(["AI is artificial intelligence."]))
        assert isinstance(result, str)

    def test_contains_query(self):
        query = "What is the boiling point of water?"
        result = build_prompt(query, make_chunks(["Water boils at 100°C."]))
        assert query in result

    def test_contains_chunk_text(self):
        chunk_text = "The sky is blue due to Rayleigh scattering."
        result = build_prompt("Why is the sky blue?", make_chunks([chunk_text]))
        assert chunk_text in result

    def test_contains_score(self):
        result = build_prompt("question", make_chunks(["some text"]))
        assert "0.90" in result

    def test_contains_context_header(self):
        result = build_prompt("question", make_chunks(["text"]))
        assert "[Context]" in result

    def test_contains_question_header(self):
        result = build_prompt("question", make_chunks(["text"]))
        assert "[Question]" in result

    def test_multiple_passages_numbered(self):
        result = build_prompt("question", make_chunks(["first", "second", "third"]))
        assert "Passage 1" in result
        assert "Passage 2" in result
        assert "Passage 3" in result

    def test_empty_query_raises(self):
        with pytest.raises(ValueError):
            build_prompt("  ", make_chunks(["text"]))

    def test_empty_chunks_raises(self):
        with pytest.raises(ValueError):
            build_prompt("question", [])

    def test_max_context_chars_limits_passages(self):
        # Each passage is ~100 chars; set budget to 150 so only 1 fits
        long_chunks = make_chunks(["x" * 100, "y" * 100, "z" * 100])
        result = build_prompt("question", long_chunks, max_context_chars=150)
        assert "Passage 2" not in result

    def test_first_passage_always_included(self):
        # Even if a single passage exceeds the budget, it must be included
        result = build_prompt("question", make_chunks(["a" * 10_000]),
                              max_context_chars=10)
        assert "Passage 1" in result

    def test_passages_in_order(self):
        result = build_prompt("q", make_chunks(["first text", "second text"]))
        assert result.index("first text") < result.index("second text")


# ---------------------------------------------------------------------------
# get_system_prompt / SYSTEM_PROMPT
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_get_system_prompt_returns_string(self):
        assert isinstance(get_system_prompt(), str)

    def test_system_prompt_not_empty(self):
        assert len(get_system_prompt()) > 0

    def test_system_prompt_constant_matches(self):
        assert get_system_prompt() == SYSTEM_PROMPT

    def test_system_prompt_mentions_context(self):
        assert "context" in get_system_prompt().lower()
