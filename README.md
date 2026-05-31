# Agent 学习课程 (Agent2L — Agent to Learning)

从零开始的 Agent（智能体）系统学习课程，系统覆盖 Agent 基础概念、大语言模型、Prompt 工程、工具系统、记忆系统、RAG、多Agent协作、评估测试、企业实践、高级技术、MCP协议、Ollama本地部署、输出解析器与LCEL等核心主题。

**本项目由AI生成，仅用于学习与研究，不涉及任何商业用途**

## 课程概览

| 章节 | 主题 | 内容概要 |
|------|------|----------|
| 第1章 | Agent基础概念 | 定义、特征、架构模式、感知-思考-行动循环、记忆系统概述 |
| 第2章 | 大语言模型基础 | Transformer架构、自注意力机制、主流模型对比、API调用 |
| 第3章 | Prompt工程与Agent设计 | CoT、ToT、ReAct、Reflexion等高级Prompt策略 |
| 第4章 | 工具系统与记忆系统 | 工具定义注册、选择执行、工作/长期/向量记忆 |
| 第5章 | 框架实践 | LangChain、LangGraph、AutoGen、CrewAI四大框架 |
| 第6章 | 高级优化技术 | Agent微调、模型蒸馏、A/B实验、Prompt版本管理 |
| 第7章 | RAG检索增强生成 | 文档分块、向量嵌入、检索策略、混合检索 |
| 第8章 | 多Agent系统 | 主从/对等/层级模式、任务分配、共识机制 |
| 第9章 | 评估与测试 | GAIA、MMLU、BIG-Bench、HELM等基准 |
| 第10章 | 前沿研究 | AI Agent技术前沿与未来方向 |
| 第11章 | 实际应用 | 真实场景案例与落地实践 |
| 第12章 | 企业级最佳实践 | 安全合规、数据隐私、成本优化、监控可观测性 |
| 第13章 | 高级技术专题 | Agent微调与LoRA/QLoRA、知识蒸馏、Prompt语义化版本管理 |
| 第14章 | MCP协议 | 模型上下文协议、Host/Client/Server架构、Tools/Resources/Prompts |
| 第15章 | Ollama本地部署 | 本地运行开源LLM、ChatOllama集成 |
| 第16章 | 输出解析器与LCEL | StrOutputParser、PydanticOutputParser、RunnableSequence/Branch/Parallel |

## 项目结构

项目采用数字编号体系，从 0 到 4 形成一个递进的学习流水线：

| 编号 | 目录 | 阶段 |
|------|------|------|
| 0 | `0-note/` | 理论学习：16章课程笔记 + 自测题 + 词汇表 |
| 1 | `1-jupyternotebook/` | 动手实践：16个配套 Jupyter Notebook |
| 2 | `2-projects/` | 项目实战：3个完整 Agent 项目 |
| 3 | `3-scaffold/` | 模板复用：Agent 脚手架，快速启动新项目 |
| 4 | `4-deploy/` | 生产部署：K8s / Docker / 监控方案 |

```
Agent2L/
├── 0-note/                          # 课程笔记与文档
│   ├── chapter1-agent-basics/       # 第1章：Agent基础概念
│   ├── chapter2-llm-fundamentals/   # 第2章：大语言模型基础
│   ├── chapter3-prompt-agent-design/ # 第3章：Prompt工程
│   ├── chapter4-tools-memory/       # 第4章：工具与记忆系统
│   ├── chapter5-framework-practice/ # 第5章：框架实践
│   ├── chapter6-advanced-optimization/ # 第6章：高级优化
│   ├── chapter7-rag-knowledge/      # 第7章：RAG检索增强
│   ├── chapter8-multi-agent-systems/ # 第8章：多Agent系统
│   ├── chapter9-evaluation-testing/ # 第9章：评估与测试
│   ├── chapter10-frontier-research/ # 第10章：前沿研究
│   ├── chapter11-practical-applications/ # 第11章：实际应用
│   ├── chapter12-enterprise-best-practices/ # 第12章：企业最佳实践
│   ├── chapter13-advanced-techniques/ # 第13章：高级技术
│   ├── chapter14-mcp-protocol/      # 第14章：MCP协议
│   ├── chapter15-ollama/            # 第15章：Ollama本地部署
│   ├── chapter16-output-parser-lcel/ # 第16章：输出解析器与LCEL
│   ├── index.md                     # 章节导航索引
│   ├── course-overview.md           # 课程总览
│   ├── glossary.md                  # 术语词汇表
│   ├── agent-learning-path-guide.md  # 学习路径指南
├── 1-jupyternotebook/               # Jupyter Notebook 配套代码
│   ├── chapter1-agent-basics.ipynb
│   ├── chapter2-llm-fundamentals.ipynb
│   └── ... (共16个章节配套Notebook)
├── projects/                        # 实战项目代码库
│   ├── smart-customer-service-agent/ # 智能客服Agent
│   ├── code-review-agent/           # 代码审查Agent
│   └── data-analysis-agent/         # 数据分析Agent
├── scaffold/                        # Agent项目脚手架模板
│   ├── agent.py / tools.py / memory.py / prompts.py
│   ├── config.py / main.py
│   └── requirements.txt / .env.example
├── 4-deploy/                        # 部署运维配置
│   ├── kubernetes/                  # K8s清单（Deployment/Service/HPA等）
│   ├── docker/                      # 生产Docker Compose
│   └── monitoring/                  # Prometheus + Grafana配置
├── .github/workflows/               # CI/CD流水线
├── requirements.txt                 # Python 依赖清单
├── Dockerfile                       # Docker 构建文件
├── docker-compose.yml               # Docker Compose 配置
├── .gitignore                       # Git 忽略规则
├── LICENSE                          # CC BY-NC 4.0 许可证
└── README.md                        # 本文件
```

## 快速开始

### 方式一：本地安装

```bash
# 1. 克隆仓库
git clone <repo-url>
cd Agent2L

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动 Jupyter Notebook
jupyter notebook
```

### 方式二：Docker 运行

```bash
# 构建并启动
docker-compose up -d

# 访问 http://localhost:8888 即可打开 Jupyter Notebook
```

## 环境变量

使用需要 API Key 的章节时，设置以下环境变量：

```bash
# OpenAI (第2、5、7章等)
export OPENAI_API_KEY=your_openai_key_here

# Anthropic (第3章等)
export ANTHROPIC_API_KEY=your_anthropic_key_here
```

或在项目根目录创建 `.env` 文件：

```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## 配套资源

- **16 章课程笔记**：每章包含详细知识点讲解、代码示例和图示
- **16 个自测题库**：每章配套选择题、判断题、简答题和实践题
- **16 个 Jupyter Notebook**：配套可运行代码示例
- **术语词汇表**：68条中英文对照核心术语（详见 [glossary.md](0-note/glossary.md)）
- **学习路径指南**：建议的学习顺序和方法（详见 [agent-learning-path-guide.md](0-note/agent-learning-path-guide.md)）
- **章节导航索引**：快速定位各章节核心内容（详见 [index.md](0-note/index.md)）

## 推荐学习路径

1. **基础入门**：第1章 → 第2章 → 第3章
2. **核心能力**：第4章 → 第7章 → 第9章
3. **框架实践**：第5章 → 第14章 → 第16章
4. **进阶提升**：第6章 → 第8章 → 第13章
5. **实战应用**：第10章 → 第11章 → 第12章
6. **本地部署**：第15章（可随时学习）

详细学习建议请参见 [agent-learning-path-guide.md](0-note/agent-learning-path-guide.md)。

## 许可证
本项目遵循 [CC BY-NC 4.0](LICENSE)，仅供学习交流使用。