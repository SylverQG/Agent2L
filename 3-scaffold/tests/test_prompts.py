import pytest

from prompts import SystemPromptBuilder, build_system_prompt


class TestSystemPromptBuilder:
    @pytest.fixture
    def builder(self):
        return SystemPromptBuilder()

    def test_add_role_returns_self(self, builder):
        result = builder.add_role("assistant")
        assert result is builder

    def test_add_role_adds_section(self, builder):
        builder.add_role("coder")
        prompt = builder.build()
        assert "You are a helpful coder." in prompt

    def test_add_instructions_adds_section(self, builder):
        builder.add_instructions("Be concise.")
        prompt = builder.build()
        assert "Be concise." in prompt

    def test_add_tools_with_names(self, builder):
        builder.add_tools(["search", "calculator"])
        prompt = builder.build()
        assert "search, calculator" in prompt

    def test_add_tools_empty_list_does_nothing(self, builder):
        builder.add_tools([])
        prompt = builder.build()
        assert prompt == ""

    def test_add_memory_note(self, builder):
        builder.add_memory_note()
        prompt = builder.build()
        assert "remember previous conversation context" in prompt

    def test_chain_all_methods(self, builder):
        prompt = (
            builder.add_role("assistant")
            .add_instructions("Answer accurately.")
            .add_tools(["search", "calculator"])
            .add_memory_note()
            .build()
        )
        assert "You are a helpful assistant." in prompt
        assert "Answer accurately." in prompt
        assert "search, calculator" in prompt
        assert "remember previous conversation context" in prompt

    def test_build_empty_builder(self, builder):
        prompt = builder.build()
        assert prompt == ""


class TestBuildSystemPrompt:
    def test_assistant_role(self):
        prompt = build_system_prompt("assistant")
        assert "helpful AI assistant" in prompt
        assert "remember previous conversation context" in prompt

    def test_researcher_role(self):
        prompt = build_system_prompt("researcher")
        assert "research assistant" in prompt

    def test_coder_role(self):
        prompt = build_system_prompt("coder")
        assert "coding assistant" in prompt

    def test_custom_role(self):
        prompt = build_system_prompt("custom")
        assert "You are an AI assistant." in prompt

    def test_with_tools(self):
        prompt = build_system_prompt("assistant", tools=["search", "calculator"])
        assert "search, calculator" in prompt

    def test_without_tools(self):
        prompt = build_system_prompt("assistant")
        assert "tools" not in prompt.lower()