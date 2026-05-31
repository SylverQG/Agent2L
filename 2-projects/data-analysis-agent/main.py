"""
数据分析 Agent 主程序。

提供命令行交互工具，加载 CSV、Excel 或 JSON 数据文件后，
用户可以通过自然语言对数据进行查询和分析，支持生成可视化图表。
"""
import os
import sys

from dotenv import load_dotenv
from analyser import DataAnalysisAgent

load_dotenv()


def main() -> None:
    """数据分析 Agent 的主入口函数。

    从环境变量读取 API 密钥，解析命令行参数获取数据文件路径，
    加载数据后进入交互式命令行循环。

    支持命令:
        - describe: 显示数据摘要统计
        - exit/quit: 退出程序
        - 其他自然语言查询: 由 AI 分析并返回结果
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY not found. "
            "Please create a .env file with OPENAI_API_KEY=your_key"
        )
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python main.py <csv_file>")
        print("Example: python main.py data.csv")
        sys.exit(1)

    filepath = sys.argv[1]

    agent = DataAnalysisAgent()
    load_result = agent.load_data(filepath)
    print(f"\n{load_result}")

    if agent.df is None:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Data Analysis Agent - Interactive Mode")
    print("  Ask questions about your data in natural language.")
    print("  Type 'describe' for data summary, 'exit' to quit.")
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
            print("Goodbye!")
            break

        if user_input.lower() == "describe":
            print("\n" + agent.describe_data())
            continue

        print("\nAnalyzing...")
        try:
            result = agent.run_query(user_input)
            print(f"\n{result}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
