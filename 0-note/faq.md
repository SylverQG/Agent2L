# 常见问题解答 (FAQ)

## 学习相关

### 需要什么基础才能学习本课程？

本课程从基础到进阶，适合有一定 Python 编程基础、对 AI 感兴趣的开发者。你需要：
- 熟悉 Python 基础语法（变量、函数、类）
- 了解基本的机器学习概念（非必须）
- 有 API 调用经验（非必须）

### 学习顺序怎么安排？

建议按章节编号顺序学习（1→16），也可以按模块跳转：

- **基础入门**：1 → 2 → 3
- **核心能力**：4 → 7 → 9
- **框架实践**：5 → 14 → 16
- **进阶提升**：6 → 8 → 13
- **实战应用**：10 → 11 → 12

详细路径请参考 [agent-learning-path-guide.md](agent-learning-path-guide.md)。

### 每章需要多长时间？

| 章节类型 | 估计时间 |
|---------|---------|
| 基础概念（ch1-4） | 每章 1-2 小时 |
| 核心技术（ch5-9） | 每章 2-3 小时 |
| 进阶专题（ch10-16） | 每章 1.5-2.5 小时 |

建议每天投入 2-3 小时，约 4-6 周完成全部课程。

---

## 环境配置

### 需要哪些 API Key？

| 服务 | 免费额度 | 用途 |
|------|---------|------|
| [OpenAI](https://platform.openai.com/api-keys) | 注册送 $5 额度 | 大部分章节的 LLM 调用 |
| [Anthropic](https://console.anthropic.com/) | 注册送 $5 额度 | 第3章 Prompt 工程（可选） |

没有 API Key 也可以：
- 学习概念和代码逻辑
- 使用第15章的 Ollama 本地模型（免费）

### 如何配置环境变量？

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=你的密钥
ANTHROPIC_API_KEY=你的密钥（可选）
```

### Docker 和本地安装哪个好？

| 方式 | 优点 | 适合场景 |
|------|------|---------|
| **本地安装** | 配置灵活、调试方便 | 个人学习、开发调试 |
| **Docker** | 环境一致、一键启动 | 多人协作、生产部署 |

详见 [README.md](../README.md) 的快速开始部分。

### pip 安装遇到问题怎么办？

```bash
# 建议使用虚拟环境
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

---

## 课程内容

### 用到了哪些框架和库？

- **LangChain** + **LangGraph**：Agent 开发核心框架
- **OpenAI** / **Anthropic**：LLM 服务提供商
- **ChromaDB** / **FAISS**：向量数据库
- **Ollama**：本地模型部署
- 完整依赖见 [requirements.txt](../requirements.txt)

### 代码示例需要修改才能运行吗？

大部分代码示例可以直接复制运行，前提是：
1. 已安装相关依赖
2. 已配置 API Key（需要 LLM 调用的章节）
3. 部分示例需要修改文件路径或 API 端点

### 自测题有答案吗？

所有自测题的答案和解析直接内嵌在题目下方，标注为 `✅ 答案` 和 `解析`。实践题附有评分标准和参考代码。

### Jupyter Notebook 怎么用？

```bash
# 方式一：Docker 运行
docker-compose up -d
# 访问 http://localhost:8888

# 方式二：本地运行
pip install -r requirements.txt
jupyter notebook
# 打开 1-jupyternotebook/ 目录下的对应文件
```

---

## 项目相关

### 实战项目需要自己准备数据吗？

| 项目 | 数据需求 |
|------|---------|
| 智能客服Agent | 内置模拟数据，直接运行 |
| 代码审查Agent | 提供自己的代码文件作为输入 |
| 数据分析Agent | 需要提供 CSV 文件 |

### 脚手架模板怎么用？

```bash
# 复制到新项目
cp -r scaffold my-project
cd my-project

# 安装并运行
pip install -r requirements.txt
cp .env.example .env
python main.py
```

### 部署运维方案有什么？

本课程提供了完整的部署方案：
- **Docker**：单机部署，适合开发和演示
- **Kubernetes**：生产级容器编排，支持自动扩缩容
- **CI/CD**：GitHub Actions 自动化流水线
- **监控**：Prometheus + Grafana

详见 [4-deploy/](../4-deploy/) 目录。

---

## 其他

### 本课程的许可证是什么？

采用 [CC BY-NC 4.0](../LICENSE) 许可证：
- **允许**：自由分享、修改（需署名）
- **禁止**：商业用途

### 如何贡献或反馈？

- 在 GitHub 上提 Issue 或 PR
- 如果有改进建议，欢迎贡献代码

### 后续会有更新吗？

本课程会持续更新，包括：
- 追踪最新的 Agent 技术发展
- 补全更多实际案例
- 优化各章节内容深度和均衡性