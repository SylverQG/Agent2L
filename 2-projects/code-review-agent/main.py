"""
代码审查 Agent 主程序。

提供命令行工具，对指定源代码文件进行自动化代码审查。
支持 Python、JavaScript、TypeScript 等语言的文件审查。
"""
import os
import sys

from dotenv import load_dotenv

from reviewers import CodeReviewer

load_dotenv()


def main() -> None:
    """代码审查工具的主入口函数。

    从环境变量读取 API 密钥，解析命令行参数获取目标文件路径，
    检查文件格式支持情况，然后调用 CodeReviewer 执行审查并输出报告。

    用法: python main.py <file_path>
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY not found. "
            "Please create a .env file with OPENAI_API_KEY=your_key"
        )
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python main.py <file_path>")
        print("Example: python main.py example.py")
        sys.exit(1)

    filepath = sys.argv[1]

    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    supported_extensions = {".py", ".js", ".jsx", ".ts", ".tsx"}
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in supported_extensions:
        print(
            f"Warning: Unsupported file extension '{ext}'. "
            f"Supported: {', '.join(supported_extensions)}"
        )
        proceed = input("Proceed anyway? (y/N): ").strip().lower()
        if proceed != "y":
            sys.exit(0)

    reviewer = CodeReviewer()
    print(f"\nAnalyzing {filepath}...\n")

    try:
        report = reviewer.review(filepath)
        print(report)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during review: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
