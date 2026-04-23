"""
Tests for conversation memory (src/memory.py).

Covers ConversationMemory: capacity, FIFO eviction, get_messages, clear.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.memory import ConversationMemory


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestConversationMemoryInit:
    def test_default_max_turns(self):
        assert ConversationMemory().max_turns == 10

    def test_custom_max_turns(self):
        assert ConversationMemory(max_turns=3).max_turns == 3

    def test_zero_max_turns_raises(self):
        with pytest.raises(ValueError):
            ConversationMemory(max_turns=0)

    def test_starts_empty(self):
        m = ConversationMemory()
        assert len(m) == 0
        assert m.is_empty

    def test_num_exchanges_zero_at_start(self):
        assert ConversationMemory().num_exchanges == 0


# ---------------------------------------------------------------------------
# add_exchange
# ---------------------------------------------------------------------------

class TestConversationMemoryAddExchange:
    def test_adds_two_turns(self):
        m = ConversationMemory()
        m.add_exchange("What is Python?", "Python is a language.")
        assert len(m) == 2

    def test_turn_roles(self):
        m = ConversationMemory()
        m.add_exchange("Q?", "A.")
        msgs = m.get_messages()
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_turn_content(self):
        m = ConversationMemory()
        m.add_exchange("Q?", "A.")
        msgs = m.get_messages()
        assert msgs[0]["content"] == "Q?"
        assert msgs[1]["content"] == "A."

    def test_multiple_exchanges_ordered(self):
        m = ConversationMemory()
        m.add_exchange("Q1", "A1")
        m.add_exchange("Q2", "A2")
        msgs = m.get_messages()
        assert msgs[0]["content"] == "Q1"
        assert msgs[2]["content"] == "Q2"

    def test_num_exchanges_increments(self):
        m = ConversationMemory()
        m.add_exchange("Q1", "A1")
        m.add_exchange("Q2", "A2")
        assert m.num_exchanges == 2

    def test_empty_question_raises(self):
        m = ConversationMemory()
        with pytest.raises(ValueError):
            m.add_exchange("   ", "A.")

    def test_empty_answer_raises(self):
        m = ConversationMemory()
        with pytest.raises(ValueError):
            m.add_exchange("Q?", "   ")

    def test_is_not_empty_after_add(self):
        m = ConversationMemory()
        m.add_exchange("Q?", "A.")
        assert not m.is_empty

    def test_evicts_oldest_exchange_at_capacity(self):
        m = ConversationMemory(max_turns=2)
        m.add_exchange("Q1", "A1")
        m.add_exchange("Q2", "A2")
        m.add_exchange("Q3", "A3")
        contents = [msg["content"] for msg in m.get_messages()]
        assert "Q1" not in contents
        assert "A1" not in contents
        assert "Q2" in contents
        assert "Q3" in contents

    def test_stays_at_max_capacity(self):
        m = ConversationMemory(max_turns=2)
        for i in range(5):
            m.add_exchange(f"Q{i}", f"A{i}")
        assert m.num_exchanges == 2
        assert len(m) == 4


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestConversationMemoryClear:
    def test_clear_removes_all_turns(self):
        m = ConversationMemory()
        m.add_exchange("Q1", "A1")
        m.add_exchange("Q2", "A2")
        m.clear()
        assert len(m) == 0
        assert m.is_empty

    def test_clear_resets_num_exchanges(self):
        m = ConversationMemory()
        m.add_exchange("Q1", "A1")
        m.clear()
        assert m.num_exchanges == 0

    def test_can_add_after_clear(self):
        m = ConversationMemory()
        m.add_exchange("Q1", "A1")
        m.clear()
        m.add_exchange("Q2", "A2")
        assert m.num_exchanges == 1

    def test_get_messages_empty_after_clear(self):
        m = ConversationMemory()
        m.add_exchange("Q1", "A1")
        m.clear()
        assert m.get_messages() == []


# ---------------------------------------------------------------------------
# get_messages
# ---------------------------------------------------------------------------

class TestConversationMemoryGetMessages:
    def test_returns_list_of_dicts(self):
        m = ConversationMemory()
        m.add_exchange("Q?", "A.")
        msgs = m.get_messages()
        assert isinstance(msgs, list)
        assert all(isinstance(d, dict) for d in msgs)

    def test_each_dict_has_role_and_content(self):
        m = ConversationMemory()
        m.add_exchange("Q?", "A.")
        for msg in m.get_messages():
            assert "role" in msg
            assert "content" in msg

    def test_empty_memory_returns_empty_list(self):
        assert ConversationMemory().get_messages() == []

    def test_does_not_mutate_internal_state(self):
        m = ConversationMemory()
        m.add_exchange("Q?", "A.")
        msgs = m.get_messages()
        msgs.clear()
        assert len(m) == 2
