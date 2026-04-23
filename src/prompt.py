"""
Prompt construction module — formats retrieved chunks and a user question
into a structured prompt ready to send to the LLM.
"""

from src.chunker import Chunk


# The system prompt instructs the model to stay grounded in the provided
# context and admit when it doesn't know, rather than hallucinating.
SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using only the context passages provided.
If the answer is not contained in the context, say "I don't have enough \
information in the provided context to answer that question."
Do not make up facts or draw on outside knowledge."""


def build_prompt(
    query: str,
    chunks: list[tuple[Chunk, float]],
    max_context_chars: int = 6000,
) -> str:
    """
    Assemble a RAG prompt from retrieved chunks and a user question.

    The prompt structure is:

        [Context]
        Passage 1 (score: 0.82):
        <text>

        Passage 2 (score: 0.74):
        <text>
        ...

        [Question]
        <query>

    Passages are included in descending score order until max_context_chars
    is reached, ensuring the prompt fits within the model's context window.

    Args:
        query: The user's question.
        chunks: List of (Chunk, score) tuples, typically from Retriever.retrieve().
        max_context_chars: Soft character budget for the context section.
                           Passages are added greedily; the first passage that
                           would exceed the budget stops the loop.

    Returns:
        Formatted prompt string ready to send as the user turn to the LLM.

    Raises:
        ValueError: If query is empty or chunks is empty.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if not chunks:
        raise ValueError("chunks must not be empty")

    context_parts: list[str] = []
    chars_used = 0

    for i, (chunk, score) in enumerate(chunks, start=1):
        passage = f"Passage {i} (score: {score:.2f}):\n{chunk.text}"
        if chars_used + len(passage) > max_context_chars and context_parts:
            # Budget exceeded — stop adding passages (always include at least one)
            break
        context_parts.append(passage)
        chars_used += len(passage)

    context_block = "\n\n".join(context_parts)

    return f"[Context]\n{context_block}\n\n[Question]\n{query}"


def get_system_prompt() -> str:
    """Return the system prompt used for all RAG queries."""
    return SYSTEM_PROMPT
