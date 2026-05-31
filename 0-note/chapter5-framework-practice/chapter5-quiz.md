# 第5章 自测题库

---

## 一、单选题（4选1）

### 1. 在LangChain框架中，以下哪个组件负责将用户输入转化为模型可处理的格式？
A. Retrieval  
B. Chains  
C. Agents  
D. Model I/O

✅ 答案：D  
**解析**：Model I/O是LangChain的核心组件之一，负责处理模型输入输出——包括Prompt模板管理、输入格式化以及模型响应的解析和处理。它是用户与模型之间的桥梁。

---

### 2. LangChain中的"Chains"组件的主要作用是什么？
A. 存储向量数据  
B. 将多个组件（Prompt、模型、工具等）串联成可执行的流水线  
C. 仅用于检索文档  
D. 替代模型推理

✅ 答案：B  
**解析**：Chains是LangChain的组合机制，允许开发者将Prompt模板、LLM调用、工具调用、输出解析器等组件按顺序或条件串联，构建可复用的处理流水线。

---

### 3. 以下关于AutoGen的描述，哪一项是正确的？
A. AutoGen是单Agent开发框架  
B. AutoGen专注于多Agent对话和协作，支持Agent之间的消息交互  
C. AutoGen只能用于代码生成任务  
D. AutoGen不支持人类参与交互

✅ 答案：B  
**解析**：AutoGen是微软推出的多Agent开发框架，核心特点是支持多个Agent通过消息进行对话和协作，支持Agent-Agent和Human-Agent等多种交互模式。

---

### 4. CrewAI框架中"Agent"的角色定义通常包含以下哪些要素？
A. 角色名（Role）和目标（Goal）  
B. 仅包含Agent的名称  
C. 仅包含Agent的内存大小  
D. 仅包含Agent的颜色主题

✅ 答案：A  
**解析**：CrewAI中定义Agent时需要指定Role（角色名，如"研究员"、"写手"）和Goal（目标描述），以及可选的Backstory（背景故事），使Agent的行为具有明确的方向性和角色特征。

---

## 二、多选题

### 5. LangChain的核心组件包括以下哪些？（多选）
A. Model I/O  
B. Retrieval  
C. Chains  
D. Agents

✅ 答案：A、B、C、D  
**解析**：LangChain的四大核心组件覆盖了从模型交互（Model I/O）、知识检索（Retrieval）、流程编排（Chains）到智能决策（Agents）的完整开发链路。

---

### 6. 以下哪些是AutoGen多Agent系统的关键特性？（多选）
A. 支持多个Agent通过消息进行对话协作  
B. 支持Human-in-the-Loop（人类参与）模式  
C. 无法进行代码执行  
D. 支持Agent角色定制和行为定义

✅ 答案：A、B、D  
**解析**：AutoGen支持多Agent对话协作、人类参与交互和Agent角色定制。同时也支持代码执行（Code Execution）能力，因此选项C错误。

---

## 三、判断题

### 7. LangChain的Retrieval组件主要用于从外部知识库中检索相关信息，以增强模型的知识覆盖范围。

✅ 答案：正确  
**解析**：Retrieval组件通过向量检索等技术，从文档、数据库等外部知识源中获取与查询相关的信息，实现RAG（检索增强生成）模式，弥补LLM的知识截止和幻觉问题。

---

### 8. CrewAI中的"Crew"指的是单个Agent独立执行的任务。

✅ 答案：错误  
**解析**：CrewAI中的"Crew"是一个Agent团队（组），由多个角色不同的Agent组成，通过分工协作共同完成复杂任务。

---

## 四、简答题

### 9. 请对比LangChain和AutoGen在Agent开发理念上的主要差异。

✅ 答案要点：  
- **LangChain**：以组件化、链式编排为核心理念，通过Chain将Prompt、模型、工具、检索等组件串联。Agent是Chains之上的决策层，负责动态选择工具和执行路径。  
- **AutoGen**：以多Agent对话协作为核心理念，将每个Agent视为独立的对话参与者，通过消息传递实现Agent间的协作和任务分发，天然支持多Agent场景。  
- **差异**：LangChain侧重组件编排和单Agent能力构建，AutoGen侧重多Agent之间的对话与协作架构。

---

### 10. 请简述在CrewAI中创建一个Agent团队（Crew）所需的核心配置步骤。

✅ 答案要点：  
- **定义Agent**：为每个Agent指定Role（角色）、Goal（目标）和Backstory（背景），定义其行为和专长。  
- **定义Task**：将总任务分解为子任务，指定每个子任务由哪个Agent执行，设置任务描述和预期输出。  
- **组建Crew**：将Agent和Task组合成Crew，配置执行流程（顺序执行或并行执行）。  
- **启动执行**：调用`crew.kickoff()`启动任务执行，监控各Agent的协作过程。