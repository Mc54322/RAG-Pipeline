"""
Generation module — sends a constructed prompt to the Anthropic API and
returns the model's response as a plain string, a streamed iterator, or
a multi-turn response with conversation history.
"""

import os
from collections.abc import Iterator

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Default model per CLAUDE.md: haiku for development, sonnet for demos.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class Generator:
    """
    Thin wrapper around the Anthropic Messages API.

    Accepts a fully-formed prompt string (built by prompt.py) and returns
    the model's text response. Keeps the API surface minimal so Day 11
    (streaming) can extend this cleanly.

    Args:
        model: Anthropic model ID. Defaults to claude-haiku-4-5-20251001
               as specified in CLAUDE.md. Swap to claude-sonnet-4-6 for
               higher-quality demo responses.
        max_tokens: Maximum tokens in the model's response.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file."
            )
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=3)

    def generate(self, prompt: str, system: str = "") -> str:
        """
        Send a prompt to the model and return its text response.

        Args:
            prompt: The user-turn message, typically built by build_prompt().
            system: Optional system prompt. Defaults to an empty string;
                    callers should pass get_system_prompt() from prompt.py.

        Returns:
            The model's response as a plain string.

        Raises:
            ValueError: If prompt is empty.
            anthropic.APIError: On any upstream API failure.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """
        Stream the model's response, yielding text chunks as they arrive.

        Uses the Anthropic streaming API so the caller can begin displaying
        output before the full response is complete — reducing perceived
        latency for end users.

        Args:
            prompt: The user-turn message, typically built by build_prompt().
            system: Optional system prompt.

        Yields:
            String chunks of the response in arrival order. Concatenating
            all chunks produces the same text as a non-streaming generate().

        Raises:
            ValueError: If prompt is empty.
            anthropic.APIError: On any upstream API failure.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    def generate_with_history(
        self,
        messages: list[dict],
        system: str = "",
    ) -> str:
        """
        Generate a response given a full multi-turn message history.

        The caller is responsible for building the messages list in
        Anthropic API format — a list of {"role": ..., "content": ...}
        dicts with alternating user/assistant turns, ending with a user
        turn. ConversationMemory.get_messages() produces this format.

        Args:
            messages: Full message history including the new user turn as
                      the last element.
            system: Optional system prompt.

        Returns:
            The model's response as a plain string.

        Raises:
            ValueError: If messages is empty or the last message is not
                        a user turn.
            anthropic.APIError: On any upstream API failure.
        """
        if not messages:
            raise ValueError("messages must not be empty")
        if messages[-1].get("role") != "user":
            raise ValueError("last message must have role 'user'")

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )

        return message.content[0].text
