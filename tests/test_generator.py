"""
Tests for the LLM generator (src/generator.py).

All tests mock the Anthropic client — no real API calls are made.
Covers generate(), stream(), and generate_with_history().
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.generator import DEFAULT_MODEL, Generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_generator() -> Generator:
    """Return a Generator with a fully mocked Anthropic client."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("src.generator.anthropic.Anthropic"):
            g = Generator()
    g.client = MagicMock()
    return g


def mock_response(text: str) -> MagicMock:
    """Minimal mock mimicking anthropic.Anthropic().messages.create() response."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


def mock_stream_cm(chunks: list[str]) -> MagicMock:
    """Context manager mock for client.messages.stream()."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.text_stream = iter(chunks)
    return cm


# ---------------------------------------------------------------------------
# Generator.generate — unit tests
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_generate_returns_string(self):
        g = make_generator()
        g.client.messages.create.return_value = mock_response("The answer.")
        assert isinstance(g.generate("What is 2+2?"), str)

    def test_generate_returns_model_text(self):
        g = make_generator()
        g.client.messages.create.return_value = mock_response("Paris.")
        assert g.generate("What is the capital of France?") == "Paris."

    def test_generate_calls_correct_model(self):
        g = make_generator()
        g.client.messages.create.return_value = mock_response("ok")
        g.generate("prompt")
        call_kwargs = g.client.messages.create.call_args[1]
        assert call_kwargs["model"] == DEFAULT_MODEL

    def test_generate_passes_system_prompt(self):
        g = make_generator()
        g.client.messages.create.return_value = mock_response("ok")
        g.generate("prompt", system="Be helpful.")
        call_kwargs = g.client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be helpful."

    def test_generate_passes_prompt_as_user_message(self):
        g = make_generator()
        g.client.messages.create.return_value = mock_response("ok")
        g.generate("my question")
        call_kwargs = g.client.messages.create.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "my question"

    def test_generate_empty_prompt_raises(self):
        g = make_generator()
        with pytest.raises(ValueError):
            g.generate("   ")

    def test_default_model_is_haiku(self):
        assert "haiku" in DEFAULT_MODEL

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(EnvironmentError):
                Generator()


# ---------------------------------------------------------------------------
# Generator.stream — unit tests
# ---------------------------------------------------------------------------

class TestGeneratorStream:
    def test_stream_yields_chunks(self):
        g = make_generator()
        g.client.messages.stream.return_value = mock_stream_cm(["Hello", " world", "!"])
        assert list(g.stream("What is Python?")) == ["Hello", " world", "!"]

    def test_stream_empty_prompt_raises(self):
        g = make_generator()
        with pytest.raises(ValueError):
            list(g.stream("   "))

    def test_stream_calls_api_with_correct_args(self):
        g = make_generator()
        g.client.messages.stream.return_value = mock_stream_cm(["answer"])
        list(g.stream("My prompt", system="My system"))
        call_kwargs = g.client.messages.stream.call_args.kwargs
        assert call_kwargs["system"] == "My system"
        assert call_kwargs["messages"] == [{"role": "user", "content": "My prompt"}]

    def test_stream_concatenated_matches_generate(self):
        """Joining stream output should match what generate() would return."""
        g = make_generator()
        chunks = ["Gradient ", "descent ", "minimises ", "the ", "loss."]
        g.client.messages.stream.return_value = mock_stream_cm(chunks)
        assert "".join(g.stream("prompt")) == "".join(chunks)


# ---------------------------------------------------------------------------
# Generator.generate_with_history — unit tests
# ---------------------------------------------------------------------------

class TestGeneratorWithHistory:
    def test_empty_messages_raises(self):
        g = make_generator()
        with pytest.raises(ValueError):
            g.generate_with_history([])

    def test_last_message_not_user_raises(self):
        g = make_generator()
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        with pytest.raises(ValueError):
            g.generate_with_history(messages)

    def test_returns_string(self):
        g = make_generator()
        g.client.messages.create.return_value = mock_response("The answer.")
        result = g.generate_with_history([{"role": "user", "content": "What is Python?"}])
        assert isinstance(result, str)
        assert result == "The answer."

    def test_passes_full_history_to_api(self):
        g = make_generator()
        g.client.messages.create.return_value = mock_response("Answer.")
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        g.generate_with_history(messages, system="sys")
        call_kwargs = g.client.messages.create.call_args.kwargs
        assert call_kwargs["messages"] == messages
        assert call_kwargs["system"] == "sys"
