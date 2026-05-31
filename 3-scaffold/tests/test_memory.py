import pytest

from memory import InMemoryConversationMemory, MemoryManager


class TestInMemoryConversationMemory:
    @pytest.fixture
    def memory(self):
        return InMemoryConversationMemory(max_turns=3)

    def test_add_message_and_get_history(self, memory):
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there")
        history = memory.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there"}

    def test_get_history_returns_copy(self, memory):
        memory.add_message("user", "Hello")
        history = memory.get_history()
        history.append({"role": "assistant", "content": "injected"})
        assert len(memory.get_history()) == 1

    def test_clear_removes_all_messages(self, memory):
        memory.add_message("user", "Hello")
        memory.clear()
        assert memory.get_history() == []

    def test_trimming_when_exceeding_max_turns(self, memory):
        for i in range(4):
            memory.add_message("user", f"Message {i}")
            memory.add_message("assistant", f"Response {i}")
        history = memory.get_history()
        assert len(history) <= 6

    def test_exact_max_turns_no_trimming(self, memory):
        for i in range(3):
            memory.add_message("user", f"Message {i}")
            memory.add_message("assistant", f"Response {i}")
        history = memory.get_history()
        assert len(history) == 6

    def test_default_max_turns(self):
        memory = InMemoryConversationMemory()
        assert memory._max_turns == 20


class TestMemoryManager:
    def test_create_default_memory_manager(self):
        manager = MemoryManager()
        assert isinstance(manager.memory, InMemoryConversationMemory)

    def test_create_memory_manager_in_memory(self):
        manager = MemoryManager(memory_type="in_memory")
        assert isinstance(manager.memory, InMemoryConversationMemory)

    def test_add_and_get_via_manager(self):
        manager = MemoryManager()
        manager.add_message("user", "Hello via manager")
        manager.add_message("assistant", "Hi via manager")
        history = manager.get_history()
        assert len(history) == 2

    def test_clear_via_manager(self):
        manager = MemoryManager()
        manager.add_message("user", "Hello")
        manager.clear()
        assert manager.get_history() == []

    def test_memory_property_returns_backend(self):
        manager = MemoryManager()
        assert manager.memory is manager._memory