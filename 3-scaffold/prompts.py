"""提示词模块 — 提供系统提示词构建工具和预定义提示模板。

支持通过 SystemPromptBuilder 链式构建自定义系统提示词，
并提供 ReAct 和 Chain-of-Thought 两种推理模式的预设提示模板。
"""

from __future__ import annotations

from typing import Literal


AgentRole = Literal["assistant", "researcher", "coder", "custom"]


class SystemPromptBuilder:
    """系统提示词构建器，支持链式调用逐步组装提示词内容。"""

    def __init__(self) -> None:
        """初始化空的提示词构建器。"""
        self._sections: list[str] = []

    def add_role(self, role: str) -> "SystemPromptBuilder":
        """添加角色描述段落。

        Args:
            role: 角色名称，如 "assistant"、"coder" 等。

        Returns:
            当前构建器实例，支持链式调用。
        """
        self._sections.append(f"You are a helpful {role}.")
        return self

    def add_instructions(self, instructions: str) -> "SystemPromptBuilder":
        """添加指令段落。

        Args:
            instructions: 具体的指令文本。

        Returns:
            当前构建器实例，支持链式调用。
        """
        self._sections.append(instructions)
        return self

    def add_tools(self, tool_names: list[str]) -> "SystemPromptBuilder":
        """添加工具列表说明段落。

        Args:
            tool_names: 可用工具的名称列表。

        Returns:
            当前构建器实例，支持链式调用。
        """
        if tool_names:
            tools_str = ", ".join(tool_names)
            self._sections.append(f"You have access to the following tools: {tools_str}.")
        return self

    def add_memory_note(self) -> "SystemPromptBuilder":
        """添加记忆能力说明段落。

        Returns:
            当前构建器实例，支持链式调用。
        """
        self._sections.append("You can remember previous conversation context.")
        return self

    def build(self) -> str:
        """将所有段落合并为最终的提示词字符串。

        Returns:
            完整的系统提示词，各段落间以空行分隔。
        """
        return "\n\n".join(self._sections)


def build_system_prompt(role: AgentRole, tools: list[str] | None = None) -> str:
    """快速构建指定角色的系统提示词。

    根据角色类型选择对应的角色描述，并可选地附加工具信息和记忆说明。

    Args:
        role: 智能体角色，可选 "assistant"、"researcher"、"coder" 或 "custom"。
        tools: 可用的工具名称列表，为 None 时不添加工具信息。

    Returns:
        构建好的系统提示词字符串。
    """
    builder = SystemPromptBuilder()

    role_map: dict[AgentRole, str] = {
        "assistant": "You are a helpful AI assistant that answers user questions accurately and concisely.",
        "researcher": "You are a research assistant. You gather information, analyze data, and provide well-sourced answers.",
        "coder": "You are a coding assistant. You write clean, well-tested code and explain technical concepts clearly.",
        "custom": "You are an AI assistant.",
    }

    builder.add_instructions(role_map.get(role, role_map["assistant"]))
    builder.add_memory_note()

    if tools:
        builder.add_tools(tools)

    return builder.build()


REACT_PROMPT = """You are an AI assistant that uses the ReAct (Reasoning + Acting) pattern.

You iterate through the following steps:
1. **Thought**: Reason about the user's question and decide what to do.
2. **Action**: Call a tool if needed to gather information.
3. **Observation**: Review the tool's output.
4. **Answer**: Provide a final answer when you have enough information.

Always respond in this format when using tools:

Thought: <your reasoning>
Action: <tool_name>(<input>)
Observation: <result>
Thought: <continue reasoning>
Answer: <final response>

When no tools are needed, just answer directly.
"""

COT_PROMPT = """You are an AI assistant that uses Chain-of-Thought reasoning.

When faced with complex questions, break down your thinking step by step:

1. Understand the question
2. Identify key components
3. Reason through each component
4. Synthesize into a final answer

Present your reasoning clearly, then give a concise final answer.
"""