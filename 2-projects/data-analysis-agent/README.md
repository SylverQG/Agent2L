# Data Analysis Agent (数据分析 Agent)

基于自然语言的数据分析助手，加载 CSV 数据后，通过对话式提问即可自动完成数据分析和可视化。

## 功能特性

- **自然语言查询**：用中文直接提问数据分析问题
- **自动代码生成**：LLM 自动生成并执行 Python 分析代码
- **数据概览**：自动输出数据集的统计摘要
- **自动绘图**：根据问题类型自动生成 matplotlib 图表
- **交互模式**：对同一数据集连续提出多个问题

## 环境配置

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=你的OpenAI密钥
```

3. 运行数据分析：

```bash
python main.py 你的数据文件.csv
```

## 使用示例

```bash
# 分析销售数据
python main.py sales_data.csv
```

加载数据后可以提问：

- "显示数据的统计摘要"
- "按地区画一个销售额条形图"
- "平均订单金额是多少？"
- "客户年龄分布是怎样的？"
- "找出销售额排名前5的产品"

输入 `exit` 或 `quit` 退出。

## 工作流程

1. 加载 CSV 文件 → 自动识别列名和数据类型
2. LLM 根据你的问题生成 pandas + matplotlib 代码
3. 自动执行生成的代码并捕获结果
4. 返回文字结果并保存图片到 `analysis_plot_N.png`