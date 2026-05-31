# 第15章：Ollama本地部署与调用 - 自测题库

---

## 一、单选题（每题5分，共6题，30分）

### 1. Ollama主要解决什么问题？
A. 云端大模型的API调用加速
B. 在本地运行开源大模型
C. 提供托管的大模型服务
D. 只用于模型训练

**✅ 答案：B**
解析：Ollama是一个专门用于在本地运行开源大模型的工具，让用户可以在自己的电脑上快速部署和使用开源模型。

### 2. Ollama本地API的默认端口是？
A. 8000
B. 8080
C. 11434
D. 5000

**✅ 答案：C**
解析：Ollama本地服务的默认监听端口是11434，这是LangChain的ChatOllama默认连接的端口。

### 3. 哪个命令可以查看本地已下载的模型？
A. ollama ls
B. ollama list
C. ollama show
D. ollama models

**✅ 答案：B**
解析：`ollama list`是查看已下载模型的正确命令；`ollama show`是查看模型详情的。

### 4. 在LangChain中用于连接Ollama的类是？
A. ChatOpenAI
B. ChatAnthropic
C. ChatOllama
D. ChatLocalModel

**✅ 答案：C**
解析：LangChain提供了`ChatOllama`类专门用于连接本地Ollama服务，使用方式和其他ChatModel类相似。

### 5. 运行Ollama本地模型时，哪个环境变量可以指定模型存储目录？
A. OLLAMA_PATH
B. OLLAMA_MODELS
C. MODEL_DIR
D. OLLAMA_CACHE

**✅ 答案：B**
解析：通过设置`OLLAMA_MODELS`环境变量，可以修改Ollama模型的存储路径。

### 6. 下面哪个命令可以运行指定模型并进入交互对话？
A. ollama pull qwen:4b
B. ollama run qwen:4b
C. ollama start qwen:4b
D. ollama launch qwen:4b

**✅ 答案：B**
解析：`ollama run model-name`是运行模型的命令，如果本地没有该模型，会先自动拉取再启动。

---

## 二、多选题（每题6分，共4题，24分）

### 1. 下面哪些是Ollama的优势？
A. 安装和使用门槛低
B. 命令简单，适合入门
C. 与LangChain的ChatOllama集成成熟
D. 本地调用通常不需要API Key
E. 永远比云端模型更强大

**✅ 答案：ABCD**
解析：Ollama的优势包括：低门槛、命令简单、LangChain集成好、本地通常不需要Key；E错误，本地模型能力通常取决于能跑多大的模型，不见得比云端大模型更强大。

### 2. 下面哪些是Ollama常用命令？
A. ollama pull
B. ollama list
C. ollama rm
D. ollama ps
E. ollama deploy

**✅ 答案：ABCD**
解析：pull（下载）、list（列出）、rm（删除）、ps（查看运行状态）都是常用命令；E不是标准命令。

### 3. Ollama vs 云端API的区别中，哪些说法正确？
A. 云端API在厂商服务器上，Ollama在本地
B. 云端API通常需要Key，Ollama本地通常不需要
C. 云端API按调用计费，Ollama本地推理主要消耗算力
D. 云端API必须联网，Ollama本地调用可以不依赖网络
E. 云端API永远比本地Ollama慢

**✅ 答案：ABCD**
解析：E错误，不一定；其他都是正确的对比。

### 4. 在LangChain中，ChatOllama可以和以下哪些组件配合使用？
A. ChatPromptTemplate
B. StrOutputParser
C. RunnableWithMessageHistory（记忆）
D. LCEL管道
E. 其他LangChain组件

**✅ 答案：ABCDE**
解析：ChatOllama作为LangChain的ChatModel实现，可以和其他标准组件完全配合，包括提示模板、解析器、记忆、LCEL等。

---

## 三、判断题（每题5分，共4题，20分）

### 1. Ollama只能在macOS和Linux上运行，不能在Windows上运行。
A. 正确
B. 错误

**✅ 答案：B**
解析：Ollama官方提供了Windows版本的下载和支持，Windows用户可以正常使用。

### 2. 使用Ollama时，模型必须先手动pull下载，不能直接run。
A. 正确
B. 错误

**✅ 答案：B**
解析：执行`ollama run`时，如果本地还没有该模型，Ollama会先自动拉取再启动，不用单独pull。

### 3. LangChain的ChatOllama和ChatOpenAI的接口风格是不同的，完全不兼容。
A. 正确
B. 错误

**✅ 答案：B**
解析：ChatOllama遵循相同的ChatModel接口，可以和其他ChatModel一样配合Prompt、Parser、LCEL使用。

### 4. Ollama的模型大小和运行速度只取决于网络带宽，和机器硬件无关。
A. 正确
B. 错误

**✅ 答案：B**
解析：模型是否跑得动、速度如何，强烈依赖机器配置，特别是内存、显存和CPU/GPU。

---

## 四、简答题（每题8分，共2题，16分）

### 1. 请简述Ollama适合在哪些场景使用，以及为什么？

**✅ 参考答案：**
适合场景：
- 本地开发/课程练习：可以快速验证想法，无需依赖网络
- 企业内网原型/隐私敏感验证：数据不出本地，可以满足安全和隐私要求
- 离线测试：在网络受限环境下也能测试
- 学习开源模型：方便在本地实验和理解不同开源模型的特性

### 2. 请描述如何在LangChain中使用ChatOllama调用本地Ollama模型？

**✅ 参考答案：**
主要步骤：
1. 确保Ollama服务已运行（默认端口11434）
2. 在Python代码中导入`ChatOllama`
3. 初始化ChatOllama时指定模型名、温度、base_url等参数
4. 和其他ChatModel一样，配合Prompt、Parser使用或构建LCEL链
5. 调用invoke()等方法执行推理

示例代码结构：
```python
from langchain_ollama import ChatOllama
model = ChatOllama(model="qwen:4b")
response = model.invoke("你好")
```

---

## 五、实践题（10分）

### 题目：本地Ollama + LangChain综合应用

**目标：**
1. 在你的电脑上安装Ollama并拉取qwen:4b（或其他小模型）
2. 使用LangChain的ChatOllama创建一个简单的翻译助手
3. 功能：输入中文，输出英文；或者反过来（自己选择）
4. 将实现代码写在一个`.py`文件中

**评分标准：**
- 代码可运行（4分）
- 使用了ChatPromptTemplate（2分）
- 使用了StrOutputParser（2分）
- 有基本的用户交互（2分）

---

## 总计得分统计

| 题型 | 满分 | 得分 |
|---|---|---|
| 单选题 | 30分 | |
| 多选题 | 24分 | |
| 判断题 | 20分 | |
| 简答题 | 16分 | |
| 实践题 | 10分 | |
| **总分** | **100分** | |

**优秀**：85分以上
**良好**：70-84分
**及格**：60-69分
**需要再学习**：60分以下
