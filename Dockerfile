FROM python:3.11-slim

WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装 Jupyter 内核
RUN python -m ipykernel install --user --name agent-course --display-name "Agent Course"

# 暴露 Jupyter 端口
EXPOSE 8888

# 默认命令：启动 Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]