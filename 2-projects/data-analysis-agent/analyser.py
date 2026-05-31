"""
数据分析 Agent 核心分析模块。

定义了 DataAnalysisAgent 类，通过大语言模型将自然语言查询
转换为 Python 代码，在隔离环境中执行并返回结果。
支持数据加载、描述统计、代码生成与执行、以及可视化图表生成。
"""
import io, os, sys, textwrap, traceback
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI


CODE_GENERATION_PROMPT = """You are a data analysis assistant. \
You have a pandas DataFrame named `df` loaded with the following columns and types:

{column_info}

First 5 rows:
{head}

Summary statistics:
{describe}

User question: {question}

Generate Python code to answer this question. The code must:
1. Use the variable `df` which is already loaded
2. If generating a plot, save it to the path: {plot_path}
3. Use plt.close() after saving to free memory
4. Output results using print() statements
5. Be complete and self-contained (only the code block)

Respond ONLY with a single Python code block enclosed in triple backticks like:
```python
import pandas as pd
import matplotlib.pyplot as plt

# your code here
print(result)
```"""


class DataAnalysisAgent:
    """数据分析 Agent，通过 LLM 驱动对 DataFrame 进行自然语言查询分析。

    支持加载多种格式的数据文件，将用户问题转化为 Python 代码执行，
    并可选地生成可视化图表保存到本地。
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        """初始化 DataAnalysisAgent 实例。

        参数:
            model: LLM 模型名称，默认为 "gpt-4o-mini"
            temperature: LLM 温度参数，控制输出的随机性，默认为 0.2
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.df: pd.DataFrame | None = None
        self.filename: str = ""
        self.plot_counter: int = 0

    def load_data(self, filepath: str) -> str:
        """从文件加载数据到 DataFrame。

        支持 CSV、Excel（.xlsx/.xls）和 JSON 格式。
        加载成功后记录文件名和数据结构信息。

        参数:
            filepath: 数据文件路径

        返回:
            加载结果文本，包含成功信息（行数列数）或错误描述
        """
        if not os.path.exists(filepath):
            return f"Error: File not found: {filepath}"

        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext == ".csv":
                self.df = pd.read_csv(filepath)
            elif ext in (".xlsx", ".xls"):
                self.df = pd.read_excel(filepath)
            elif ext == ".json":
                self.df = pd.read_json(filepath)
            else:
                return f"Error: Unsupported file format: {ext}"
        except Exception as e:
            return f"Error loading file: {e}"

        self.filename = os.path.basename(filepath)
        rows, cols = self.df.shape
        cols_str = ', '.join(self.df.columns.tolist())
        return (
            f"Successfully loaded '{self.filename}': {rows} rows × {cols} columns"
            f"\n\nColumns: {cols_str}"
        )

    def _get_column_info(self) -> str:
        """获取 DataFrame 各列的数据类型和空值信息。

        返回:
            格式化后的列信息文本，每行列名、类型和空值数量
        """
        if self.df is None:
            return ""
        info: list[str] = []
        for col in self.df.columns:
            dtype = self.df[col].dtype
            nulls = self.df[col].isnull().sum()
            info.append(f"  - {col}: {dtype} ({nulls} null values)")
        return "\n".join(info)

    def describe_data(self) -> str:
        """生成数据集的详细描述统计信息。

        包括 DataFrame 形状、各列数据类型、空值数量
        以及 describe() 输出的描述性统计。

        返回:
            格式化后的数据描述文本，如果未加载数据则返回提示
        """
        if self.df is None:
            return "No data loaded."

        desc = self.df.describe(include="all").to_string()
        null_counts = self.df.isnull().sum().to_string()
        dtypes = self.df.dtypes.to_string()

        return (
            f"DataFrame Info:\n"
            f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n\n"
            f"Data Types:\n{dtypes}\n\n"
            f"Null Counts:\n{null_counts}\n\n"
            f"Descriptive Statistics:\n{desc}"
        )

    def _get_plot_path(self) -> str:
        """生成下一个可视化图表的保存路径。

        使用自增计数器生成唯一的 PNG 文件名。

        返回:
            图表的绝对路径字符串
        """
        self.plot_counter += 1
        return os.path.abspath(f"analysis_plot_{self.plot_counter}.png")

    def run_query(self, question: str) -> str:
        """对数据集运行自然语言查询。

        将问题与数据集信息组装成提示词发送给 LLM，
        LLM 生成对应的 Python 分析代码后在隔离环境中执行。

        参数:
            question: 用户的自然语言查询

        返回:
            查询结果文本，包含原始问题、执行结果及可视化保存路径
        """
        if self.df is None:
            return "No data loaded. Please load a dataset first."

        column_info = self._get_column_info()
        head = self.df.head().to_string()
        describe = self.df.describe(include="all").to_string()
        plot_path = self._get_plot_path()

        messages = [
            SystemMessage(
                content="You are a data analysis expert. Output ONLY Python code."
            ),
            HumanMessage(
                content=CODE_GENERATION_PROMPT.format(
                    column_info=column_info,
                    head=head,
                    describe=describe,
                    question=question,
                    plot_path=plot_path,
                )
            ),
        ]

        response = self.llm.invoke(messages)
        code = self._extract_code(response.content.strip())

        if code is None:
            return (
                "I couldn't generate analysis code. "
                "Please try rephrasing your question."
            )

        result_text = self._execute_code(code)
        plot_result = ""

        if os.path.exists(plot_path):
            plot_result = f"\n\nVisualization saved to: {plot_path}"

        return f"Query: {question}\n\n{result_text}{plot_result}"

    def _extract_code(self, text: str) -> str | None:
        """从 LLM 返回文本中提取 Python 代码块。

        优先匹配 ```python ... ``` 格式的代码块，
        如果未匹配到则尝试检测是否包含 import 和 print/plt 等代码特征。

        参数:
            text: LLM 返回的文本

        返回:
            提取到的 Python 代码字符串，未找到时返回 None
        """
        import re

        match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        if "import" in text and ("print(" in text or "plt." in text):
            return text.strip()

        return None

    def _execute_code(self, code: str) -> str:
        """在隔离环境中执行生成的 Python 分析代码。

        将 DataFrame 和 pandas/matplotlib 注入局部命名空间，
        重定向标准输出以捕获 print 结果，执行完成后恢复。

        参数:
            code: 要执行的 Python 代码字符串

        返回:
            代码执行的输出文本，发生异常时返回错误堆栈信息
        """
        local_vars: dict[str, Any] = {
            "df": self.df,
            "pd": pd,
            "plt": plt,
        }

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            compiled = compile(textwrap.dedent(code), "<analysis>", "exec")
            exec(compiled, local_vars)
            output = sys.stdout.getvalue().strip()
            return output if output else "Analysis completed successfully."
        except Exception:
            tb = traceback.format_exc()
            return f"Error during analysis:\n{tb}"
        finally:
            sys.stdout = old_stdout
