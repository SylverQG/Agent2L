"""Agent 模块 — 提供 Agent、ReActAgent 和 ToolUsingAgent 三种智能体实现。

该模块封装了与大语言模型（LLM）的交互逻辑，支持：
- 基础对话智能体（Agent）
- 思考-行动（ReAct）智能体
- 工具调用智能体（ToolUsingAgent）
- 自动根据环境配置选择 LLM 后端（OpenAI / Anthropic / Ollama）
"""

from __future__ import annotations

from typing import Any, Callable
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from config import config
from memory import MemoryManager
from tools import ToolRegistry


class Agent:
    """基础智能体，封装 LLM 调用、工具注册和对话记忆管理。

    支持通过配置自动选择 LLM 后端，并提供 run 方法执行任务。
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        tools: ToolRegistry | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        model_name: str | None = None,
    ) -> None:
        """初始化智能体。

        Args:
            llm: 大语言模型实例，未提供时自动创建默认模型。
            tools: 工具注册表，未提供时创建空注册表。
            system_prompt: 系统提示词，定义智能体的行为角色。
            model_name: 指定使用的模型名称，优先级高于配置中的默认模型。
        """
        self.llm = llm or self._create_default_llm(model_name)
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.memory_manager: MemoryManager = MemoryManager("in_memory")

    def _create_default_llm(self, model_name: str | None = None) -> BaseChatModel:
        """根据环境配置自动创建默认的大语言模型实例。

        按优先级依次尝试：OpenAI → Anthropic → Ollama（本地回退）。

        Args:
            model_name: 可选的模型名称，覆盖配置中的 default_model。

        Returns:
            配置好的 BaseChatModel 实例。
        """
        model = model_name or config.default_model

        if config.openai_api_key:
            return ChatOpenAI(
                model=model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.openai_api_key,
                base_url=config.openai_api_base or None,
            )

        if config.anthropic_api_key:
            return ChatAnthropic(
                model=model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.anthropic_api_key,
            )

        return ChatOllama(
            model=model,
            temperature=config.temperature,
            base_url=config.ollama_base_url,
        )

    def add_tool(self, tool_fn: Callable[..., Any]) -> None:
        """向智能体注册一个新工具。

        Args:
            tool_fn: 要注册的工具函数。
        """
        self.tools.register(tool_fn)

    def set_memory(self, memory_type: str) -> None:
        """切换智能体的记忆类型。

        Args:
            memory_type: 记忆类型，如 "in_memory" 或 "vector"。
        """
        self.memory_manager = MemoryManager(memory_type)

    def _build_messages(self, task: str) -> list:
        """构建发送给 LLM 的消息列表，包含系统提示词、历史对话和当前任务。

        Args:
            task: 当前用户输入的任务。

        Returns:
            消息列表，供 LLM invoke 使用。
        """
        messages: list = [SystemMessage(content=self.system_prompt)]

        for msg in self.memory_manager.get_history():
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(HumanMessage(content=msg["content"]))

        messages.append(HumanMessage(content=task))
        return messages

    def run(self, task: str) -> str:
        """执行给定的任务并返回 LLM 的响应。

        自动管理对话历史记录。

        Args:
            task: 用户输入的任务描述或问题。

        Returns:
            LLM 生成的响应文本。
        """
        self.memory_manager.add_message("user", task)

        messages = self._build_messages(task)
        response = self.llm.invoke(messages)

        result = response.content if hasattr(response, "content") else str(response)
        if isinstance(result, list):
            result = " ".join(str(r) for r in result)

        self.memory_manager.add_message("assistant", result)
        return result


class ReActAgent(Agent):
    """基于 ReAct（推理+行动）模式的智能体。

    将可用工具信息注入系统提示词，引导 LLM 按「思考→行动→观察→回答」的
    循环模式进行推理和工具调用。
    """

    def run(self, task: str) -> str:
        """使用 ReAct 模式执行任务，在提示词中注入可用工具列表。

        Args:
            task: 用户输入的任务描述或问题。

        Returns:
            LLM 生成的响应文本。
        """
        tool_names = [t['name'] for t in self.tools.list_tools()]
        prompt = (
            f"{self.system_prompt}\n\nUser: {task}\n\n"
            f"Available tools: {tool_names}"
        )
        self.memory_manager.add_message("user", task)

        messages = self._build_messages(task)
        messages[0] = SystemMessage(content=prompt)

        response = self.llm.invoke(messages)
        result = response.content if hasattr(response, "content") else str(response)
        if isinstance(result, list):
            result = " ".join(str(r) for r in result)

        self.memory_manager.add_message("assistant", result)
        return result


class ToolUsingAgent(Agent):
    """支持 LangChain bind_tools 机制的工具调用智能体。

    将工具注册表中的工具绑定到 LLM，使模型能够原生感知并调用工具。
    使用惰性初始化，仅在首次 run 时绑定工具。
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        tools: ToolRegistry | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> None:
        """初始化 ToolUsingAgent。

        Args:
            llm: 大语言模型实例。
            tools: 工具注册表。
            system_prompt: 系统提示词。
        """
        super().__init__(llm, tools, system_prompt)
        self._bound_llm: BaseChatModel | None = None

    def run(self, task: str) -> str:
        """使用工具绑定的 LLM 执行任务，支持模型原生工具调用。

        Args:
            task: 用户输入的任务描述或问题。

        Returns:
            LLM 生成的响应文本。
        """
        self.memory_manager.add_message("user", task)

        if self._bound_llm is None:
            langchain_tools = self.tools.to_langchain_list()
            try:
                self._bound_llm = self.llm.bind_tools(langchain_tools)
            except (AttributeError, TypeError, NotImplementedError):
                self._bound_llm = self.llm

        messages = self._build_messages(task)
        response = self._bound_llm.invoke(messages)

        result = response.content if hasattr(response, "content") else str(response)
        if isinstance(result, list):
            result = " ".join(str(r) for r in result)

        self.memory_manager.add_message("assistant", result)
        return result
