# 第16章 自测题库

---

## 一、单选题（共4题）

### 1. 在LangChain中，输出解析器的主要作用是什么？
A. 将用户输入转换为LLM可理解的格式  
B. 将LLM的文本输出转换为结构化数据  
C. 提高LLM的生成速度  
D. 减少LLM的token消耗

✅ **答案：B**  
**解析：** 输出解析器负责将LLM生成的非结构化文本转换为可被程序处理的结构化数据，如JSON、Pydantic对象等。

---

### 2. 以下哪个解析器最适合用于强类型验证的场景？
A. `StrOutputParser`  
B. `JsonOutputParser`  
C. `PydanticOutputParser`  
D. `CommaSeparatedListOutputParser`

✅ **答案：C**  
**解析：** `PydanticOutputParser` 结合Pydantic模型，提供了强大的类型验证功能，可以确保输出符合预定义的数据结构和类型约束。

---

### 3. 在LCEL中，管道符`|`的作用是什么？
A. 注释代码  
B. 组合Runnable组件  
C. 导入模块  
D. 定义变量

✅ **答案：B**  
**解析：** LCEL使用管道符`|`来组合各个Runnable组件，形成处理链，实现声明式的工作流编排。

---

### 4. 以下哪个组件用于条件分支处理？
A. `RunnableSequence`  
B. `RunnableParallel`  
C. `RunnableBranch`  
D. `RunnableLambda`

✅ **答案：C**  
**解析：** `RunnableBranch` 允许根据条件动态选择不同的处理分支，实现条件路由逻辑。

---

## 二、多选题（共3题）

### 5. 以下哪些是常见的输出解析器类型？（多选）
A. `JsonOutputParser`  
B. `PydanticOutputParser`  
C. `DatetimeOutputParser`  
D. `EnumOutputParser`

✅ **答案：A、B、C、D**  
**解析：** LangChain提供了多种输出解析器，包括JSON解析、Pydantic模型解析、日期时间解析、枚举解析等，以满足不同的结构化输出需求。

---

### 6. 以下哪些是Runnable接口的核心方法？（多选）
A. `invoke()`  
B. `batch()`  
C. `stream()`  
D. `execute()`

✅ **答案：A、B、C**  
**解析：** Runnable接口的核心方法包括`invoke()`（单次调用）、`batch()`（批量调用）、`stream()`（流式输出），以及对应的异步版本`ainvoke()`、`abatch()`、`astream()`。

---

### 7. 以下关于LCEL的描述，哪些是正确的？（多选）
A. LCEL是LangChain Expression Language的缩写  
B. 使用`|`操作符组合组件  
C. 支持流式输出  
D. 只支持同步调用

✅ **答案：A、B、C**  
**解析：** LCEL支持同步和异步调用，因此选项D错误。其他选项都是LCEL的正确描述。

---

## 三、判断题（共3题）

### 8. `StrOutputParser` 是最简单的输出解析器，它直接返回LLM输出的字符串。

✅ **答案：正确**  
**解析：** `StrOutputParser` 是最基础的解析器，不做任何结构化处理，直接返回LLM的原始文本输出。

---

### 9. `RunnableParallel` 用于顺序执行多个Runnable组件。

✅ **答案：错误**  
**解析：** `RunnableParallel` 用于并行执行多个组件，而`RunnableSequence`才是用于顺序执行。

---

### 10. 使用`get_format_instructions()` 方法可以获取提示词，告诉LLM应该如何格式化输出。

✅ **答案：正确**  
**解析：** 所有输出解析器都实现了`get_format_instructions()`方法，该方法生成格式化指导的提示词，帮助LLM生成符合预期格式的输出。

---

## 四、简答题（共2题）

### 11. 请简述`JsonOutputParser`和`PydanticOutputParser`的主要区别。

✅ **答案要点：**  
- **`JsonOutputParser`**：将输出解析为Python字典，适合简单的结构化需求，可以配合JSON Schema使用，但类型验证能力相对较弱。  
- **`PydanticOutputParser`**：将输出解析为Pydantic模型对象，提供了强大的类型验证、字段约束和数据验证功能，适合需要严格类型安全的场景。  
- **适用场景**：简单的JSON解析用前者，需要强类型验证和复杂数据结构用后者。

---

### 12. 请简述LCEL中`RunnableSequence`、`RunnableParallel`、`RunnableBranch`和`RunnableLambda`的各自用途。

✅ **答案要点：**  
- **`RunnableSequence`**：顺序执行多个组件，前一个的输出作为后一个的输入，使用`|`操作符创建。  
- **`RunnableParallel`**：并行执行多个组件，同时运行并返回一个包含所有结果的字典。  
- **`RunnableBranch`**：条件分支，根据输入的条件选择不同的处理分支。  
- **`RunnableLambda`**：将自定义的Python函数包装为Runnable，方便在LCEL管道中使用自定义逻辑。

---

## 五、实践题（共1题）

### 13. 请使用LCEL构建一个简单的文本分析管道，要求：
1. 输入一段文本
2. 并行执行以下任务：
   - 提取文本的关键词（5个）
   - 分析情感（正面/负面/中性）
   - 生成摘要（50字以内）
3. 输出包含这三个结果的JSON格式

（提示：使用`RunnableParallel`、`ChatPromptTemplate`、`JsonOutputParser`等组件）

✅ **参考代码：**
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, CommaSeparatedListOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List

# 初始化LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# 定义输出模型
class AnalysisResult(BaseModel):
    keywords: List[str] = Field(description="关键词列表")
    sentiment: str = Field(description="情感倾向")
    summary: str = Field(description="文本摘要")

# 创建各个任务的提示词
keyword_prompt = ChatPromptTemplate.from_template(
    "提取以下文本的5个关键词，用逗号分隔：{text}"
)
sentiment_prompt = ChatPromptTemplate.from_template(
    "分析以下文本的情感，只返回'正面'、'负面'或'中性'：{text}"
)
summary_prompt = ChatPromptTemplate.from_template(
    "用50字以内总结以下文本：{text}"
)

# 创建并行处理管道
analysis_chain = RunnableParallel({
    "keywords": keyword_prompt | llm | CommaSeparatedListOutputParser(),
    "sentiment": sentiment_prompt | llm | StrOutputParser(),
    "summary": summary_prompt | llm | StrOutputParser()
})

# 测试
text = "LangChain提供了强大的输出解析器和LCEL，让开发者可以轻松构建结构化的LLM应用。"
result = analysis_chain.invoke({"text": text})
print(result)
```

---

**总分：100分**
- 单选题：每题10分，共40分
- 多选题：每题10分，共30分  
- 判断题：每题5分，共15分
- 简答题：每题7分，共14分
- 实践题：1分（仅作为拓展练习）

**实践题评分标准：**

| 评分维度 | 分值 | 要求 |
|---------|------|------|
| RunnableParallel 使用 | 2分 | 正确创建并行管道，三个子任务同时执行 |
| 输出解析器选择 | 2分 | 关键词使用 `CommaSeparatedListOutputParser` 或 `JsonOutputParser` |
| 代码可运行性 | 3分 | 代码逻辑完整，无语法错误，可正常执行 |
| 结果格式 | 2分 | 输出为结构化格式（JSON/dict），三个结果完整 |
| 扩展性 | 1分 | 管道设计可扩展，便于添加新的分析任务 |
