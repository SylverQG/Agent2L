# 章节导航索引

> 快速定位各章节核心知识点与代码示例

---

## 第1章：Agent基础概念

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| Agent定义与特征 | [正文](chapter1-agent-basics/chapter1-agent-basics.md) 第1节 | [Notebook](../../1-jupyternotebook/chapter1-agent-basics.ipynb) |
| 感知-思考-行动循环 | [正文](chapter1-agent-basics/chapter1-agent-basics.md) 第2节 | `感知-思考-行动` 循环实现 |
| Agent架构模式 | [正文](chapter1-agent-basics/chapter1-agent-basics.md) 第3节 | 单Agent vs 多Agent对比 |
| 记忆系统概述 | [正文](chapter1-agent-basics/chapter1-agent-basics.md) 第4节 | 工作记忆/长期记忆/向量记忆 |
| 自测题 | [自测题](chapter1-agent-basics/chapter1-quiz.md) | 选择题+判断题+简答题 |

## 第2章：大语言模型基础

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| Transformer架构 | [正文](chapter2-llm-fundamentals/chapter2-llm-fundamentals.md) 第1节 | 注意力机制可视化 |
| 自注意力机制 | [正文](chapter2-llm-fundamentals/chapter2-llm-fundamentals.md) 第2节 | 简化版注意力实现 |
| 主流模型对比 | [正文](chapter2-llm-fundamentals/chapter2-llm-fundamentals.md) 第3节 | GPT-4/Claude/Gemini/LLaMA对比 |
| API调用示例 | [正文](chapter2-llm-fundamentals/chapter2-llm-fundamentals.md) 第4节 | OpenAI/Anthropic API调用 |
| 自测题 | [自测题](chapter2-llm-fundamentals/chapter2-quiz.md) | 选择题+判断题+简答题 |

## 第3章：Prompt工程与Agent设计

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| Prompt设计原则 | [正文](chapter3-prompt-agent-design/chapter3-prompt-agent-design.md) 第1节 | 角色/任务/格式/约束设计 |
| Chain-of-Thought | [正文](chapter3-prompt-agent-design/chapter3-prompt-agent-design.md) 第2节 | CoT实现与Few-shot CoT |
| Tree of Thoughts | [正文](chapter3-prompt-agent-design/chapter3-prompt-agent-design.md) 第3节 | ToT多路径推理 |
| ReAct模式 | [正文](chapter3-prompt-agent-design/chapter3-prompt-agent-design.md) 第4节 | 推理-行动循环实现 |
| Reflexion | [正文](chapter3-prompt-agent-design/chapter3-prompt-agent-design.md) 第5节 | 自我反思机制 |
| 自测题 | [自测题](chapter3-prompt-agent-design/chapter3-quiz.md) | 选择题+多选题+判断题+简答题 |

## 第4章：工具系统与记忆系统

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| 工具定义与注册 | [正文](chapter4-tools-memory/chapter4-tools-memory.md) 第1节 | `@tool` 装饰器与BaseTool |
| 工具选择与执行 | [正文](chapter4-tools-memory/chapter4-tools-memory.md) 第2节 | 顺序/并行/条件执行策略 |
| 工作记忆 | [正文](chapter4-tools-memory/chapter4-tools-memory.md) 第3节 | ConversationBufferMemory |
| 长期记忆 | [正文](chapter4-tools-memory/chapter4-tools-memory.md) 第4节 | 文件存储/SQLite持久化 |
| 向量记忆 | [正文](chapter4-tools-memory/chapter4-tools-memory.md) 第5节 | ChromaDB向量检索 |
| 知识图谱记忆 | [正文](chapter4-tools-memory/chapter4-tools-memory.md) 第6节 | NetworkX图结构记忆 |
| 自测题 | [自测题](chapter4-tools-memory/chapter4-quiz.md) | 选择题+判断题+简答题 |

## 第5章：框架实践

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| LangChain核心 | [正文](chapter5-framework-practice/chapter5-framework-practice.md) 第1节 | Chain/LCEL/Runnable构建 |
| LangGraph | [正文](chapter5-framework-practice/chapter5-framework-practice.md) 第2节 | 状态图/条件边/循环 |
| AutoGen | [正文](chapter5-framework-practice/chapter5-framework-practice.md) 第3节 | 多Agent对话/代码执行 |
| CrewAI | [正文](chapter5-framework-practice/chapter5-framework-practice.md) 第4节 | 角色分工/任务委派 |
| 自测题 | [自测题](chapter5-framework-practice/chapter5-quiz.md) | 选择题+多选题+判断题+简答题 |

## 第6章：高级优化技术

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| Agent微调 | [正文](chapter6-advanced-optimization/chapter6-advanced-optimization.md) 第1节 | Fine-tuning数据集/LoRA/QLoRA |
| 模型蒸馏 | [正文](chapter6-advanced-optimization/chapter6-advanced-optimization.md) 第2节 | 知识蒸馏/温度参数/Alpha参数 |
| A/B实验框架 | [正文](chapter6-advanced-optimization/chapter6-advanced-optimization.md) 第3节 | 统计显著性/流量分配 |
| Prompt版本管理 | [正文](chapter6-advanced-optimization/chapter6-advanced-optimization.md) 第4节 | 语义化版本/Canary发布 |
| 自测题 | [自测题](chapter6-advanced-optimization/chapter6-quiz.md) | 选择题+多选题+判断题+简答题 |

## 第7章：RAG检索增强生成

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| RAG架构概述 | [正文](chapter7-rag-knowledge/chapter7-rag-knowledge.md) 第1节 | 索引-检索-生成流程 |
| 文档分块策略 | [正文](chapter7-rag-knowledge/chapter7-rag-knowledge.md) 第2节 | 固定大小/语义分块/层级分块 |
| 向量嵌入 | [正文](chapter7-rag-knowledge/chapter7-rag-knowledge.md) 第3节 | OpenAI/自定义嵌入模型 |
| 检索策略 | [正文](chapter7-rag-knowledge/chapter7-rag-knowledge.md) 第4节 | 相似度搜索/MMR/混合检索 |
| 自测题 | [自测题](chapter7-rag-knowledge/chapter7-quiz.md) | 选择题+多选题+判断题+简答题 |

## 第8章：多Agent系统

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| 多Agent架构模式 | [正文](chapter8-multi-agent-systems/chapter8-multi-agent-systems.md) 第1节 | 主从/对等/层级模式 |
| 任务分配策略 | [正文](chapter8-multi-agent-systems/chapter8-multi-agent-systems.md) 第2节 | 集中式/分散式/拍卖式 |
| 通信与协调 | [正文](chapter8-multi-agent-systems/chapter8-multi-agent-systems.md) 第3节 | 消息传递/共享黑板 |
| 共识机制 | [正文](chapter8-multi-agent-systems/chapter8-multi-agent-systems.md) 第4节 | 投票/辩论/加权决策 |
| 自测题 | [自测题](chapter8-multi-agent-systems/chapter8-quiz.md) | 选择题+多选题+判断题+简答题 |

## 第9章：评估与测试

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| 基准测试体系 | [正文](chapter9-evaluation-testing/chapter9-evaluation-testing.md) 第1节 | GAIA/MMLU/BIG-Bench/HELM |
| 自动评估方法 | [正文](chapter9-evaluation-testing/chapter9-evaluation-testing.md) 第2节 | LLM-as-Judge/BLEU/ROUGE |
| 人工评估 | [正文](chapter9-evaluation-testing/chapter9-evaluation-testing.md) 第3节 | 评估维度/评分标准 |
| 持续测试流水线 | [正文](chapter9-evaluation-testing/chapter9-evaluation-testing.md) 第4节 | CI集成/回归测试 |
| 自测题 | [自测题](chapter9-evaluation-testing/chapter9-quiz.md) | 选择题+多选题+判断题+简答题 |

## 第10章：前沿研究

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| 自主Agent前沿 | [正文](chapter10-frontier-research/chapter10-frontier-research.md) 第1节 | 浏览器/编程/科研Agent |
| 多模态Agent | [正文](chapter10-frontier-research/chapter10-frontier-research.md) 第2节 | 视觉/语音/代码多模态 |
| Agent安全与对齐 | [正文](chapter10-frontier-research/chapter10-frontier-research.md) 第3节 | 奖励黑客/越狱攻击/RLHF |
| 自测题 | [自测题](chapter10-frontier-research/chapter10-quiz.md) | 选择题+多选题+判断题+简答题+实践题 |

## 第11章：实际应用

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| 客服Agent | [正文](chapter11-practical-applications/chapter11-practical-applications.md) 第1节 | 意图识别/知识库检索/工单创建 |
| 编程助手 | [正文](chapter11-practical-applications/chapter11-practical-applications.md) 第2节 | 代码生成/调试/审查 |
| 数据分析Agent | [正文](chapter11-practical-applications/chapter11-practical-applications.md) 第3节 | SQL查询/可视化/报告生成 |
| 自测题 | [自测题](chapter11-practical-applications/chapter11-quiz.md) | 选择题+多选题+判断题+简答题+实践题 |

## 第12章：企业级最佳实践

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| 安全与合规 | [正文](chapter12-enterprise-best-practices/chapter12-enterprise-best-practices.md) 第1节 | 数据隐私/内容过滤/审计日志 |
| 成本优化 | [正文](chapter12-enterprise-best-practices/chapter12-enterprise-best-practices.md) 第2节 | Token优化/模型选择/缓存策略 |
| 性能调优 | [正文](chapter12-enterprise-best-practices/chapter12-enterprise-best-practices.md) 第3节 | 并发处理/异步优化/负载均衡 |
| 监控与可观测性 | [正文](chapter12-enterprise-best-practices/chapter12-enterprise-best-practices.md) 第4节 | 日志链路追踪/性能指标/告警 |
| 自测题 | [自测题](chapter12-enterprise-best-practices/chapter12-quiz.md) | 选择题+多选题+判断题+简答题+实践题 |

## 第13章：高级技术专题

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| Agent微调 | [正文](chapter13-advanced-techniques/chapter13-advanced-techniques.md) 第1节 | LoRA配置/QLoRA量化/微调流程 |
| 知识蒸馏 | [正文](chapter13-advanced-techniques/chapter13-advanced-techniques.md) 第2节 | 温度参数/T2T蒸馏/T5蒸馏 |
| Prompt版本管理 | [正文](chapter13-advanced-techniques/chapter13-advanced-techniques.md) 第3节 | 语义化版本/Canary发布/回滚策略 |
| 自测题 | [自测题](chapter13-advanced-techniques/chapter13-quiz.md) | 选择题+多选题+判断题+简答题+实践题 |

## 第14章：MCP协议

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| MCP架构概述 | [正文](chapter14-mcp-protocol/chapter14-mcp-protocol.md) 第1节 | Host/Client/Server三层架构 |
| Tools能力 | [正文](chapter14-mcp-protocol/chapter14-mcp-protocol.md) 第2节 | 工具定义/调用/错误处理 |
| Resources能力 | [正文](chapter14-mcp-protocol/chapter14-mcp-protocol.md) 第3节 | 资源URI/订阅/通知 |
| Prompts能力 | [正文](chapter14-mcp-protocol/chapter14-mcp-protocol.md) 第4节 | Prompt模板/动态参数 |
| 传输层 | [正文](chapter14-mcp-protocol/chapter14-mcp-protocol.md) 第5节 | stdio/Streamable HTTP |
| 自测题 | [自测题](chapter14-mcp-protocol/chapter14-quiz.md) | 选择题+判断题+简答题 |

## 第15章：Ollama本地部署

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| Ollama安装与配置 | [正文](chapter15-ollama/chapter15-ollama.md) 第1节 | 模型拉取/运行/管理 |
| LangChain集成 | [正文](chapter15-ollama/chapter15-ollama.md) 第2节 | ChatOllama/OllamaEmbeddings |
| 本地RAG应用 | [正文](chapter15-ollama/chapter15-ollama.md) 第3节 | 文档加载/向量存储/检索生成 |
| 带记忆的对话 | [正文](chapter15-ollama/chapter15-ollama.md) 第4节 | 会话记忆/消息历史管理 |
| 自测题 | [自测题](chapter15-ollama/chapter15-quiz.md) | 选择题+判断题+简答题+实践题 |

## 第16章：输出解析器与LCEL

| 知识点 | 位置 | 代码/Notebook |
|--------|------|---------------|
| 输出解析器 | [正文](chapter16-output-parser-lcel/chapter16-output-parser-lcel.md) 第1节 | StrOutputParser/JsonOutputParser/PydanticOutputParser |
| 结构化输出 | [正文](chapter16-output-parser-lcel/chapter16-output-parser-lcel.md) 第2节 | TypedDict/Pydantic/JSON Schema |
| Runnable组合 | [正文](chapter16-output-parser-lcel/chapter16-output-parser-lcel.md) 第3节 | RunnableSequence/RunnableBranch/RunnableParallel/RunnableLambda |
| 流式处理 | [正文](chapter16-output-parser-lcel/chapter16-output-parser-lcel.md) 第4节 | 事件流/异步迭代/自定义回调 |
| 自测题 | [自测题](chapter16-output-parser-lcel/chapter16-quiz.md) | 选择题+判断题+简答题+实践题 |

---

## 附录

| 资源 | 位置 |
|------|------|
| 术语词汇表 | [glossary.md](glossary.md) |
| 课程总览 | [course-overview.md](course-overview.md) |
| 学习路径指南 | [agent-learning-path-guide.md](agent-learning-path-guide.md) |
| 依赖清单 | [requirements.txt](../requirements.txt) |
| Docker环境 | [Dockerfile](../Dockerfile) / [docker-compose.yml](../docker-compose.yml) |
| 实战项目 | [2-projects/](../2-projects)（智能客服/代码审查/数据分析） |
| 脚手架模板 | [3-scaffold/](../3-scaffold)（快速初始化Agent项目） |
| 部署运维 | [4-deploy/](../4-deploy)（K8s/CI/CD/监控） |