"""
智能客服 Agent 主程序。

提供基于 LangChain 与大语言模型的智能客服交互功能。
支持知识库检索、创建工单和查询订单状态三种工具。
"""
import os
import sys
import json
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from tools import search_knowledge_base, create_ticket, check_order_status

load_dotenv()

SYSTEM_PROMPT = """You are a smart customer service assistant. Your role is to help customers with their inquiries.

You have access to the following tools:
1. search_knowledge_base(query: str) - Search the knowledge base for answers
2. create_ticket(customer_name: str, issue: str) - Create a support ticket
3. check_order_status(order_id: str) - Check the status of an order

To use a tool, you MUST respond with a JSON block in the following format:
{"tool": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}

After receiving the tool result, provide a natural language response to the customer.

If the customer greets you, greet them back without using any tool.
If the customer is saying goodbye, say goodbye warmly.
If you cannot determine what the customer needs, ask clarifying questions.

Always be polite, professional, and helpful."""

TOOL_MAP: dict[str, Any] = {
    "search_knowledge_base": search_knowledge_base,
    "create_ticket": create_ticket,
    "check_order_status": check_order_status,
}


def extract_tool_call(text: str) -> dict[str, Any] | None:
    """从 LLM 返回的文本中提取工具调用 JSON 块。

    在文本中查找第一个 "{" 和最后一个 "}"，尝试将其解析为 JSON，
    并检查是否包含 "tool" 字段。

    参数:
        text: LLM 返回的原始响应文本

    返回:
        如果解析成功且包含 tool 字段，返回工具调用字典；否则返回 None
    """
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        block = text[start:end]
        parsed = json.loads(block)
        if "tool" in parsed:
            return parsed
    except (ValueError, json.JSONDecodeError):
        pass
    return None


def run_agent(user_input: str, llm: ChatOpenAI, messages: list) -> str:
    """运行智能客服 Agent 的单轮对话。

    将用户输入添加到消息列表，调用 LLM 获取响应。
    如果 LLM 返回了工具调用指令，则执行对应工具并将结果反馈给 LLM
    以生成最终的自然语言回复。

    参数:
        user_input: 用户的输入文本
        llm: LangChain ChatOpenAI 实例
        messages: 对话历史消息列表

    返回:
        最终回复文本（直接回复或经过工具调用后的回复）
    """
    messages.append(HumanMessage(content=user_input))

    response = llm.invoke(messages)
    assistant_reply = response.content.strip()
    messages.append(AIMessage(content=assistant_reply))

    tool_call = extract_tool_call(assistant_reply)
    if tool_call is not None:
        tool_name = tool_call["tool"]
        arguments = tool_call.get("arguments", {})

        if tool_name not in TOOL_MAP:
            error_msg = f"Unknown tool: {tool_name}. Available tools: {', '.join(TOOL_MAP.keys())}"
            messages.append(SystemMessage(content=error_msg))
            return error_msg

        try:
            tool_fn = TOOL_MAP[tool_name]
            result = tool_fn(**arguments)
        except TypeError as e:
            result = f"Error calling {tool_name}: {e}"

        messages.append(SystemMessage(content=f"Tool result: {result}"))

        final_response = llm.invoke(messages)
        final_text = final_response.content.strip()
        messages.append(AIMessage(content=final_text))
        return final_text

    return assistant_reply


def main() -> None:
    """智能客服 Agent 的主入口函数。

    从环境变量读取 API 密钥，初始化 LLM 和消息列表，
    随后进入交互式命令行循环，持续接收用户输入并调用 run_agent 处理。

    支持输入 "exit" 或 "quit" 退出程序。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY not found. "
            "Please create a .env file with OPENAI_API_KEY=your_key"
        )
        sys.exit(1)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    print("=" * 60)
    print("  Smart Customer Service Agent")
    print("  Type 'exit' or 'quit' to end the session")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("\nAgent: Thank you for contacting us. Have a great day!")
            break

        print("\nAgent: ", end="", flush=True)
        try:
            reply = run_agent(user_input, llm, messages)
            print(reply)
        except Exception as e:
            print(f"I apologize, but I encountered an error: {e}")
            messages.pop()
            messages.pop()


if __name__ == "__main__":
    main()