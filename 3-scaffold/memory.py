"""记忆模块 — 提供智能体的对话历史存储与管理功能。

支持两种记忆后端：
- InMemoryConversationMemory：基于内存的对话历史存储，固定最大轮数。
- VectorMemory：基于 ChromaDB 的向量化记忆存储，支持语义检索。

MemoryManager 作为统一入口，负责切换不同记忆后端。
"""

from __future__ import annotations

from typing import Any, Protocol


class Memory(Protocol):
    """记忆接口协议，定义对话记忆的基本操作。"""

    def add_message(self, role: str, content: str) -> None:
        """添加一条对话消息。

        Args:
            role: 消息角色，如 "user" 或 "assistant"。
            content: 消息内容。
        """

    def get_history(self) -> list[dict[str, str]]:
        """获取完整的对话历史。

        Returns:
            包含 role 和 content 的字典列表。
        """

    def clear(self) -> None:
        """清除所有对话历史。"""


class InMemoryConversationMemory:
    """基于内存的对话记忆实现，按最大轮数自动裁剪历史。"""

    def __init__(self, max_turns: int = 20) -> None:
        """初始化内存对话记忆。

        Args:
            max_turns: 保留的最大对话轮数（每轮含用户和助理两条消息）。
        """
        self._messages: list[dict[str, str]] = []
        self._max_turns = max_turns

    def add_message(self, role: str, content: str) -> None:
        """添加一条消息，超出最大轮数时自动裁剪最早的消息。

        Args:
            role: 消息角色（"user" 或 "assistant"）。
            content: 消息文本内容。
        """
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self._max_turns * 2:
            self._messages = self._messages[-(self._max_turns * 2):]

    def get_history(self) -> list[dict[str, str]]:
        """返回对话历史的副本。"""
        return list(self._messages)

    def clear(self) -> None:
        """清除所有对话历史。"""
        self._messages.clear()


class VectorMemory:
    """基于 ChromaDB 的向量化记忆实现，支持语义检索。

    当 chromadb 不可用时，自动回退为纯内存存储，query 方法返回提示信息。
    """

    def __init__(self, collection_name: str = "agent_memory") -> None:
        """初始化向量记忆，尝试连接 ChromaDB。

        Args:
            collection_name: ChromaDB 集合名称。
        """
        self._collection_name = collection_name
        self._messages: list[dict[str, str]] = []
        try:
            import chromadb
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(collection_name)
            self._using_chromadb = True
        except (ImportError, Exception):
            self._using_chromadb = False

    def add_message(self, role: str, content: str) -> None:
        """添加消息到内存，同时若 ChromaDB 可用则同步写入向量库。

        Args:
            role: 消息角色。
            content: 消息内容。
        """
        self._messages.append({"role": role, "content": content})
        if self._using_chromadb:
            idx = str(len(self._messages) - 1)
            self._collection.add(
                documents=[f"{role}: {content}"],
                ids=[idx],
            )

    def get_history(self) -> list[dict[str, str]]:
        """返回完整的对话历史。"""
        return list(self._messages)

    def query(self, query_text: str, top_k: int = 3) -> list[str]:
        """对历史消息执行语义搜索，返回最相关的结果。

        Args:
            query_text: 查询文本。
            top_k: 返回的最相关结果数量。

        Returns:
            匹配的文档内容列表。若 ChromaDB 不可用则返回提示信息。
        """
        if not self._using_chromadb:
            return ["(Vector search unavailable — chromadb not available)"]
        results = self._collection.query(query_texts=[query_text], n_results=top_k)
        return results.get("documents", [[]])[0]

    def clear(self) -> None:
        """清除内存中的历史记录，并从 ChromaDB 中删除集合。"""
        self._messages.clear()
        if self._using_chromadb:
            self._client.delete_collection(self._collection_name)


class MemoryManager:
    """记忆管理器，统一封装对记忆后端的访问并支持运行时切换。"""

    def __init__(self, memory_type: str = "in_memory") -> None:
        """初始化记忆管理器。

        Args:
            memory_type: 记忆类型，"in_memory"（默认）或 "vector"。
        """
        self._memory: InMemoryConversationMemory | VectorMemory
        if memory_type == "vector":
            self._memory = VectorMemory()
        else:
            self._memory = InMemoryConversationMemory()

    @property
    def memory(self) -> InMemoryConversationMemory | VectorMemory:
        """返回当前使用的记忆后端实例。"""
        return self._memory

    def switch(self, memory_type: str) -> None:
        """切换到指定类型的记忆后端（会丢失当前会话历史）。

        Args:
            memory_type: 目标记忆类型，"in_memory" 或 "vector"。
        """
        if memory_type == "vector":
            self._memory = VectorMemory()
        else:
            self._memory = InMemoryConversationMemory()

    def add_message(self, role: str, content: str) -> None:
        """添加消息到当前记忆后端。"""
        self._memory.add_message(role, content)

    def get_history(self) -> list[dict[str, str]]:
        """获取当前记忆后端的对话历史。"""
        return self._memory.get_history()

    def clear(self) -> None:
        """清除当前记忆后端的所有历史记录。"""
        self._memory.clear()