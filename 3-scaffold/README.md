# Agent 项目脚手架

基于 LangChain 的 Agent 项目快速启动模板，帮助你快速创建新的 Agent 项目。

## 快速开始

```bash
# 1. 复制脚手架到新项目
cp -r scaffold my-agent-project
cd my-agent-project

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 4. 运行示例
python main.py              # 基础对话模式
python main.py react        # ReAct 推理-行动模式
python main.py tooluse      # 工具绑定模式
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `config.py` | 配置管理 — 使用 Pydantic Settings 从 `.env` 加载配置 |
| `agent.py` | Agent 基类 — 支持 OpenAI / Anthropic / Ollama 三种 LLM 后端 |
| `tools.py` | 工具管理 — `@tool` 装饰器示例 + `ToolRegistry` 注册管理器 |
| `memory.py` | 记忆系统 — 对话记忆 + 向量记忆（ChromaDB 存根） |
| `prompts.py` | 提示词构建器 — 支持助手/研究员/程序员等角色模板 |
| `main.py` | 程序入口 — 交互式 CLI 对话框 |
| `requirements.txt` | Python 依赖清单 |
| `.env.example` | 环境变量模板 |

## 自定义指南

1. **Agent 行为** — 继承 `agent.py` 中的 `Agent` 基类，重写 `run()` 方法
2. **添加工具** — 在 `tools.py` 中用 `@tool` 装饰器定义新函数，通过 `ToolRegistry` 注册
3. **切换记忆** — 可将 `InMemoryConversationMemory` 替换为 `VectorMemory` 实现持久化
4. **自定义提示词** — 使用 `SystemPromptBuilder` 构建角色专属的系统提示词

## 环境要求

- Python 3.11+
- 至少一个 LLM 提供商的 API Key（OpenAI / Anthropic / Ollama）