"""主入口模块 — 提供命令行交互式智能体聊天程序。

支持三种智能体模式（base / react / tooluse），
通过命令行参数选择，并自动检测 API 密钥配置。
"""

from __future__ import annotations

import sys

from agent import Agent, ReActAgent, ToolUsingAgent
from config import config
from prompts import REACT_PROMPT, build_system_prompt
from tools import registry


def check_api_key() -> bool:
    """检查 API 密钥配置状态，若均未配置则给出回退提示。

    Returns:
        是否至少有一个 API 密钥已配置。
    """
    if not any([config.openai_api_key, config.anthropic_api_key]):
        print("=" * 60)
        print("WARNING: No API keys found!")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file.")
        print("Falling back to Ollama (ensure it's running on localhost:11434).")
        print("=" * 60)
        return False
    return True


def interactive_chat(agent: Agent) -> None:
    """启动交互式聊天循环，处理用户输入并调用智能体。

    支持特殊命令：
    - 'quit'：退出程序
    - 'clear'：清除对话记忆
    - 'tools'：列出已注册的工具

    Args:
        agent: 要使用的智能体实例。
    """
    print("\nInteractive Agent Chat")
    print("Type 'quit' to exit, 'clear' to clear memory, 'tools' to list tools.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            agent.memory_manager.clear()
            print("[Memory cleared]\n")
            continue

        if user_input.lower() == "tools":
            print("Registered tools:")
            for t in registry.list_tools():
                print(f"  - {t['name']}: {t['description']}")
            print()
            continue

        response = agent.run(user_input)
        print(f"Agent: {response}\n")


def main() -> None:
    """程序入口：解析命令行参数，初始化智能体并启动交互式聊天。

    命令行参数指定智能体类型：
    - （无参数）或 "base"：基础 Agent
    - "react"：ReActAgent
    - "tooluse"：ToolUsingAgent
    """
    check_api_key()

    agent_type = "base"
    if len(sys.argv) > 1:
        agent_type = sys.argv[1].lower()

    agent: Agent

    if agent_type == "react":
        system_prompt = REACT_PROMPT
        agent = ReActAgent(tools=registry, system_prompt=system_prompt)
        print("[ReActAgent mode]")
    elif agent_type == "tooluse":
        tool_names = [t["name"] for t in registry.list_tools()]
        system_prompt = build_system_prompt("assistant", tool_names)
        agent = ToolUsingAgent(tools=registry, system_prompt=system_prompt)
        print("[ToolUsingAgent mode]")
    else:
        system_prompt = build_system_prompt("assistant")
        agent = Agent(tools=registry, system_prompt=system_prompt)
        print("[Base Agent mode]")

    print(f"Model: {config.default_model}")
    print(f"Temperature: {config.temperature}")
    print(f"Max tokens: {config.max_tokens}")

    interactive_chat(agent)


if __name__ == "__main__":
    main()
