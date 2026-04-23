"""
Conversation memory module — stores the history of a multi-turn dialogue
so the model can understand follow-up questions.

Without memory, every query is independent. With memory, the model can
answer "What else did Watson discover?" by looking back at the previous
turn where Watson and Crick were discussed.

Memory format
─────────────
History is stored as a list of Anthropic-compatible message dicts:

    [
        {"role": "user",      "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user",      "content": "follow-up question"},
        ...
    ]

This format is passed directly to the Anthropic Messages API alongside
the new user turn, giving the model full conversational context.

Why store questions (not RAG prompts) in history
─────────────────────────────────────────────────
The full RAG prompt for each turn includes retrieved passages, which can
be thousands of characters. Storing these in the history would quickly
exhaust the model's context window. Instead, history stores only the bare
question and the model's answer — compact, readable, and sufficient for
the model to understand what has been discussed.

The current turn's RAG context (fresh retrieved passages) is always
appended as the final user message, so the model gets relevant context
for the new question even when older context is not re-sent.
"""

from dataclasses import dataclass, field


@dataclass
class Turn:
    """A single message in a conversation."""

    role: str     # "user" or "assistant"
    content: str


class ConversationMemory:
    """
    Fixed-capacity conversation history for a single dialogue session.

    Stores up to max_turns complete exchanges (each exchange = one user
    message + one assistant message). When the limit is exceeded, the
    oldest exchange is dropped to make room for the new one.

    Args:
        max_turns: Maximum number of (user, assistant) pairs to retain.
                   Older exchanges are evicted FIFO when the limit is hit.
                   Default 10 keeps ~20 messages, comfortably within
                   Claude's context window for typical query lengths.
    """

    def __init__(self, max_turns: int = 10) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be at least 1, got {max_turns}")
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def add_exchange(self, question: str, answer: str) -> None:
        """
        Record one complete dialogue exchange.

        Appends a user turn (the bare question) followed by an assistant
        turn (the generated answer). If storing this exchange would exceed
        max_turns, the oldest exchange is evicted first.

        Args:
            question: The user's question (stored without RAG context).
            answer: The model's answer for this turn.
        """
        if not question.strip():
            raise ValueError("question must not be empty")
        if not answer.strip():
            raise ValueError("answer must not be empty")

        # Evict the oldest exchange (2 turns) if at capacity
        if len(self._turns) >= self.max_turns * 2:
            self._turns = self._turns[2:]

        self._turns.append(Turn(role="user", content=question))
        self._turns.append(Turn(role="assistant", content=answer))

    def clear(self) -> None:
        """Remove all stored turns, resetting to a fresh session."""
        self._turns = []

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_messages(self) -> list[dict]:
        """
        Return the conversation history in Anthropic API message format.

        The returned list can be passed directly to the `messages` argument
        of `client.messages.create()`, typically with a new user turn
        appended at the end.

        Returns:
            List of {"role": ..., "content": ...} dicts, oldest first.
        """
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def __len__(self) -> int:
        """Return the total number of individual turns (not exchanges)."""
        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        """True if no turns have been recorded yet."""
        return len(self._turns) == 0

    @property
    def num_exchanges(self) -> int:
        """Return the number of complete (user, assistant) pairs stored."""
        return len(self._turns) // 2
