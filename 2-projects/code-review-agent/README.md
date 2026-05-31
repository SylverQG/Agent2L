# Code Review Agent (代码审查 Agent)

基于 LLM 的自动化代码审查工具，自动分析源代码的质量、安全、风格和性能问题。

## 功能特性

- **多语言支持**：支持 Python、JavaScript、TypeScript
- **四个审查维度**：代码质量、安全性、代码风格、性能
- **严重级别分级**：严重、主要、次要、建议
- **完整报告输出**：包含问题行号、详细描述和修复建议

## 环境配置

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=你的OpenAI密钥
```

3. 运行审查：

```bash
python main.py 你要审查的文件路径
```

## 使用示例

```bash
# 审查 Python 文件
python main.py example.py

# 审查 JavaScript 文件
python main.py app.js

# 审查 TypeScript 文件
python main.py component.tsx
```

## 审查报告结构

审查完成后输出格式化报告，包含以下信息：
- 问题描述和严重级别
- 所在行号
- 问题类别（质量/安全/风格/性能）
- 具体修复建议