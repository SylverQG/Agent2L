"""
代码审查 Agent 数据模型与审查逻辑模块。

定义了代码审查结果的数据结构（Finding、ReviewReport）
以及核心审查器 CodeReviewer，调用 LLM 对源代码进行分析并生成审查报告。
"""
from dataclasses import dataclass, field
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

Severity = Literal["critical", "major", "minor", "suggestion"]
Category = Literal["quality", "security", "style", "performance"]

LANGUAGE_HINTS: dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript (React JSX)",
    ".ts": "TypeScript",
    ".tsx": "TypeScript (React TSX)",
}


@dataclass
class Finding:
    """表示代码审查中发现的一个问题。

    包含问题类别、严重级别、所在行号、描述和修复建议。
    """
    category: Category
    severity: Severity
    line: int
    description: str
    suggestion: str


@dataclass
class ReviewReport:
    """代码审查报告，包含被审查文件信息和所有发现的问题。

    提供按类别汇总问题数量的 summary 属性，以及格式化的报告字符串输出。
    """
    filename: str
    language: str
    total_lines: int
    findings: list[Finding] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, int]:
        """按类别统计问题数量。

        返回:
            字典，键为问题类别（quality/security/style/performance），值为对应数量
        """
        counts: dict[str, int] = {"quality": 0, "security": 0, "style": 0, "performance": 0}
        for f in self.findings:
            counts[f.category] += 1
        return counts

    def __str__(self) -> str:
        """生成格式化的代码审查报告字符串。

        报告包含文件信息、按类别汇总的问题数量、以及按严重级别排序
        的每个问题的详细描述和修复建议。

        返回:
            格式化后的多行报告文本
        """
        lines = [
            "=" * 60,
            f"  Code Review Report",
            f"  File: {self.filename}",
            f"  Language: {self.language}",
            f"  Total Lines: {self.total_lines}",
            f"  Total Findings: {len(self.findings)}",
            "=" * 60,
        ]
        summary = self.summary
        lines.append("")
        lines.append("Summary by Category:")
        for cat, count in summary.items():
            lines.append(f"  - {cat.capitalize()}: {count}")
        lines.append("")

        if not self.findings:
            lines.append("No issues found. Great code!")
            return "\n".join(lines)

        lines.append("Detailed Findings:")
        lines.append("-" * 60)

        severity_order = {"critical": 0, "major": 1, "minor": 2, "suggestion": 3}
        sorted_findings = sorted(self.findings, key=lambda f: (severity_order.get(f.severity, 99), f.line))

        for i, finding in enumerate(sorted_findings, 1):
            lines.append(f"")
            lines.append(f"  #{i} [{finding.severity.upper()}] [{finding.category.capitalize()}]")
            lines.append(f"  Line {finding.line}: {finding.description}")
            lines.append(f"  Suggestion: {finding.suggestion}")

        return "\n".join(lines)


class CodeReviewer:
    """使用大语言模型对源代码进行自动化代码审查。

    通过向 LLM 发送代码内容和审查指令，获取结构化的审查结果，
    并将 LLM 返回的 JSON 解析为 Finding 对象列表。
    """
    REVIEW_PROMPT = """You are a senior code reviewer. Analyze the provided code and identify issues in these categories:
- quality: Code quality issues (logic errors, dead code, anti-patterns, etc.)
- security: Security vulnerabilities (injection, XSS, hardcoded secrets, etc.)
- style: Style issues (naming conventions, formatting, etc.)
- performance: Performance issues (inefficient loops, unnecessary allocations, etc.)

For each finding, assign a severity: critical, major, minor, or suggestion.

You MUST respond ONLY with a valid JSON array. Each element must have these fields:
- category: one of "quality", "security", "style", "performance"
- severity: one of "critical", "major", "minor", "suggestion"
- line: integer line number
- description: brief description of the issue
- suggestion: how to fix it

If no issues found, return an empty array [].

Code to review:
```{language}
{code}
```"""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        """初始化 CodeReviewer 实例。

        参数:
            model: LLM 模型名称，默认为 "gpt-4o-mini"
            temperature: LLM 温度参数，控制输出的随机性，默认为 0.3
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def _detect_language(self, filepath: str) -> str:
        """根据文件扩展名检测编程语言。

        参数:
            filepath: 文件路径

        返回:
            对应的语言名称字符串，未知扩展名返回 "Unknown"
        """
        import os
        ext = os.path.splitext(filepath)[1].lower()
        return LANGUAGE_HINTS.get(ext, "Unknown")

    def review(self, filepath: str) -> ReviewReport:
        """对指定源代码文件执行代码审查。

        读取文件内容，检测语言，调用 LLM 进行分析，
        并将结果解析为结构化的 ReviewReport 对象。

        参数:
            filepath: 待审查的源代码文件路径

        返回:
            包含所有审查结果的 ReviewReport 对象

        异常:
            FileNotFoundError: 文件不存在时抛出
        """
        import os

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()

        language = self._detect_language(filepath)
        total_lines = code.count("\n") + 1

        messages = [
            SystemMessage(content="You are a senior code reviewer. Output ONLY valid JSON, no other text."),
            HumanMessage(
                content=self.REVIEW_PROMPT.format(language=language, code=code)
            ),
        ]

        response = self.llm.invoke(messages)
        raw = response.content.strip()

        findings = self._parse_findings(raw)
        return ReviewReport(
            filename=os.path.basename(filepath),
            language=language,
            total_lines=total_lines,
            findings=findings,
        )

    def _parse_findings(self, raw: str) -> list[Finding]:
        """解析 LLM 返回的 JSON 文本为 Finding 对象列表。

        尝试直接解析 JSON，若失败则通过正则提取 JSON 数组再尝试。
        对每个条目进行字段验证和默认值处理。

        参数:
            raw: LLM 返回的原始文本

        返回:
            Finding 对象列表，解析失败时返回空列表
        """
        import json
        import re

        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("Response is not a list")
        except (json.JSONDecodeError, ValueError):
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        findings: list[Finding] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            category = item.get("category", "quality")
            severity = item.get("severity", "suggestion")
            line = item.get("line", 0)
            description = item.get("description", "")
            suggestion = item.get("suggestion", "")
            if category not in ("quality", "security", "style", "performance"):
                category = "quality"
            if severity not in ("critical", "major", "minor", "suggestion"):
                severity = "suggestion"
            findings.append(
                Finding(
                    category=category,
                    severity=severity,
                    line=line,
                    description=description,
                    suggestion=suggestion,
                )
            )

        return findings