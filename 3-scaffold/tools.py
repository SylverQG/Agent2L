"""工具模块 — 提供工具注册管理与内置工具集合。

支持工具的注册、注销、查询和批量选取，并集成了 LangChain 工具装饰器
以便与 LangChain 生态兼容。内置工具包括：网络搜索、计算器和日期时间。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from langchain_core.tools import tool as langchain_tool


class ToolRegistry:
    """工具注册表，管理一组可被智能体调用的工具函数。

    提供注册、注销、查询、列表化以及批量选择等操作，同时支持
    转换为 LangChain 工具格式以便与 LangChain 集成。
    """

    def __init__(self) -> None:
        """初始化空的工具注册表。"""
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """注册一个工具函数，以其函数名作为键。

        Args:
            fn: 要注册的工具函数。

        Returns:
            注册的原始函数。
        """
        name = getattr(fn, "__name__", None) or getattr(fn, "name", None) or str(fn)
        self._tools[name] = fn
        return fn

    def unregister(self, name: str) -> None:
        """注销一个已注册的工具。

        Args:
            name: 要注销的工具名称（函数名）。
        """
        self._tools.pop(name, None)

    def get(self, name: str) -> Callable[..., Any] | None:
        """根据名称获取已注册的工具函数。

        Args:
            name: 工具名称。

        Returns:
            工具函数，若不存在则返回 None。
        """
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """列出所有已注册的工具及其文档描述。

        Returns:
            包含名称和描述的字典列表。
        """
        return [
            {"name": name, "description": fn.__doc__ or ""}
            for name, fn in self._tools.items()
        ]

    def to_langchain_list(self) -> list[Callable[..., Any]]:
        """将所有已注册的工具转换为 LangChain 工具列表。

        Returns:
            LangChain 兼容的工具函数列表。
        """
        return list(self._tools.values())

    def select(self, names: list[str]) -> list[Callable[..., Any]]:
        """按名称批量选取工具，若某个工具不存在则抛出 KeyError。

        Args:
            names: 要选取的工具名称列表。

        Returns:
            选取的工具函数列表。

        Raises:
            KeyError: 当指定名称的工具未注册时抛出。
        """
        result = {}
        for name in names:
            fn = self._tools.get(name)
            if fn:
                result[name] = fn
            else:
                msg = f"Tool '{name}' not found in registry"
                raise KeyError(msg)
        return list(result.values())


registry = ToolRegistry()


@langchain_tool
def web_search(query: str) -> str:
    """搜索网络获取当前信息。当前返回占位结果，需接入 SerpAPI / Tavily 等实现。"""
    return f'[web_search] Results for "{query}": (placeholder — implement with SerpAPI / Tavily)'


@langchain_tool
def calculator(expression: str) -> str:
    """计算数学表达式。使用 Python 兼容语法，如 "2 + 3 * 4"。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@langchain_tool
def current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """获取当前日期时间，按指定格式返回。"""
    return datetime.now().strftime(format)


registry.register(web_search)
registry.register(calculator)
registry.register(current_datetime)