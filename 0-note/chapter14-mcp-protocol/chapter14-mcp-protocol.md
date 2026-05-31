
# 第14章：MCP模型上下文协议

## 📖 章节概述

本章将深入学习MCP（Model Context Protocol，模型上下文协议），这是一个用于连接AI助手与外部数据源和工具的开放协议。你将了解MCP的核心概念、架构设计、传输方式，以及如何在LangChain中集成和使用MCP，最终通过完整的代码示例掌握MCP的实际应用。

**学习时长**：2-3周  
**难度等级**：⭐⭐⭐ 中高级  
**核心技能**：理解MCP协议、搭建MCP服务器和客户端、与LangChain集成

---

## 14.1 MCP协议简介

### 14.1.1 什么是MCP？

**MCP（Model Context Protocol）** 是由Anthropic提出的开放协议，旨在为AI助手提供标准化的方式来访问外部数据源和工具。它定义了一套统一的接口规范，使得AI助手能够安全、高效地与各种服务和数据进行交互。

```
┌─────────────────────────────────────────────────────────┐
│                      MCP协议层                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   AI助手     │◄──►│   MCP客户端   │◄──►│  MCP服务  │ │
│  │（Claude等）  │    │              │    │    器     │ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
│                                  ▲                      │
│                                  │                      │
│                       ┌──────────┴──────────┐          │
│                       │                     │          │
│                  ┌────▼────┐         ┌──────▼─────┐    │
│                  │  工具   │         │  资源      │    │
│                  └─────────┘         └────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

```mermaid
flowchart TB
    A[🤖 AI助手] &lt;--&gt; B[📡 MCP客户端]
    B &lt;--&gt; C[🔗 MCP传输层]
    C &lt;--&gt; D[⚙️ MCP服务器]
    D &lt;--&gt; E[🛠️ 工具]
    D &lt;--&gt; F[📚 资源]
    D &lt;--&gt; G[💬 提示模板]
    style A fill:#f97316,color:#fff
    style D fill:#3b82f6,color:#fff
```

### 14.1.2 MCP解决的问题

MCP协议主要解决以下核心问题：

#### 1. **工具和资源的标准化访问**
- 提供统一的接口规范，无需为每个工具单独开发集成
- 支持自动发现和描述能力，AI助手可以了解可用功能

#### 2. **安全性和权限控制**
- 提供细粒度的权限管理机制
- 支持用户授权和审计追踪

#### 3. **上下文管理**
- 标准化的上下文传递机制
- 支持流式和批量数据传输

#### 4. **生态系统互操作性**
- 开放协议，促进生态系统发展
- 支持多语言、多平台实现

### 14.1.3 MCP与其他技术的区别

| 特性 | MCP | 传统工具调用 | RAG系统 | Agent框架 |
|------|-----|-------------|---------|----------|
| **标准化程度** | 高度标准化 | 自定义实现 | 各有差异 | 框架特定 |
| **安全性** | 内置安全机制 | 需要自行实现 | 较弱 | 取决于框架 |
| **发现能力** | 自动发现 | 手动配置 | 需配置 | 需配置 |
| **资源支持** | 原生支持 | 有限 | 主要是文档 | 通常不支持 |
| **生态系统** | 开放生态 | 分散 | 多样 | 各有生态 |

---

## 14.2 MCP架构与核心能力

### 14.2.1 MCP的三层架构

MCP协议采用客户端-服务器架构，主要包含三个核心组件：

#### 1. **MCP客户端（Client）**
- 运行在AI助手一侧
- 负责与服务器通信
- 处理协议消息的序列化和反序列化

#### 2. **MCP服务器（Server）**
- 提供具体的功能实现
- 暴露工具、资源和提示模板
- 处理客户端请求并返回结果

#### 3. **传输层（Transport）**
- 负责客户端和服务器之间的通信
- 支持多种传输方式（stdio、HTTP等）
- 处理消息的路由和交付

（详见 [第5章 - 框架实践](chapter5-framework-practice/chapter5-framework-practice.md)）

### 14.2.2 MCP核心能力

MCP提供三大核心能力：

#### 🔧 **工具（Tools）**
工具是MCP服务器提供的可执行函数，AI助手可以调用这些工具来执行具体操作。

```python
# MCP工具定义示例
{
    "name": "search_web",
    "description": "搜索网络获取最新信息",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数量",
                "default": 5
            }
        },
        "required": ["query"]
    }
}
```

#### 📚 **资源（Resources）**
资源是MCP服务器提供的数据源，可以是文件、数据库记录、API响应等。

```python
# MCP资源定义示例
{
    "uri": "file:///docs/report.pdf",
    "name": "年度报告",
    "description": "公司2024年度财务报告",
    "mimeType": "application/pdf"
}
```

#### 💬 **提示模板（Prompts）**
提示模板是预定义的提示词模板，可以帮助AI助手更好地完成特定任务。

```python
# MCP提示模板示例
{
    "name": "code_review",
    "description": "代码审查提示模板",
    "arguments": [
        {
            "name": "code",
            "description": "要审查的代码",
            "required": true
        }
    ]
}
```
（详见 [第4章 - 工具与记忆系统](chapter4-tools-memory/chapter4-tools-memory.md)）

---

## 14.3 MCP传输方式

### 14.3.1 stdio传输

stdio（标准输入输出）是最简单的传输方式，适用于本地进程间通信。

```python
# stdio传输示例
import sys
import json

class StdioTransport:
    def __init__(self):
        self.running = False
    
    def start(self):
        self.running = True
        while self.running:
            try:
                # 从stdin读取消息
                line = sys.stdin.readline()
                if not line:
                    break
                
                message = json.loads(line.strip())
                self.handle_message(message)
                
            except Exception as e:
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    }
                }), flush=True)
    
    def send_message(self, message):
        print(json.dumps(message), flush=True)
    
    def handle_message(self, message):
        # 处理消息
        pass
```

### 14.3.2 Streamable HTTP传输

Streamable HTTP传输适用于网络环境，支持流式数据传输。

```python
# HTTP传输服务端示例
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import asyncio

app = FastAPI()

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    async def generate():
        # 处理流式请求
        async for line in request.stream():
            message = json.loads(line)
            response = await handle_request(message)
            yield json.dumps(response) + "\n"
    
    return StreamingResponse(generate(), media_type="application/json")

async def handle_request(message):
    # 处理MCP请求
    return {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "result": {"status": "ok"}
    }
```

---

## 14.4 MCP协议规范

### 14.4.1 JSON-RPC 2.0基础

MCP使用JSON-RPC 2.0作为基础协议，所有消息都是JSON格式。

#### 请求消息格式
```json
{
    "jsonrpc": "2.0",
    "id": "unique-request-id",
    "method": "method_name",
    "params": {}
}
```

#### 响应消息格式
```json
{
    "jsonrpc": "2.0",
    "id": "unique-request-id",
    "result": {}
}
```

#### 错误消息格式
```json
{
    "jsonrpc": "2.0",
    "id": "unique-request-id",
    "error": {
        "code": -32601,
        "message": "Method not found"
    }
}
```

### 14.4.2 核心方法

MCP定义了一系列核心方法：

#### 1. 初始化（initialize）
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {
            "name": "my-client",
            "version": "1.0.0"
        }
    }
}
```

#### 2. 列出工具（tools/list）
```json
{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
}
```

#### 3. 调用工具（tools/call）
```json
{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "search_web",
        "arguments": {
            "query": "MCP协议",
            "limit": 5
        }
    }
}
```

#### 4. 列出资源（resources/list）
```json
{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "resources/list",
    "params": {}
}
```

#### 5. 读取资源（resources/read）
```json
{
    "jsonrpc": "2.0",
    "id": 5,
    "method": "resources/read",
    "params": {
        "uri": "file:///docs/report.pdf"
    }
}
```

---

## 14.5 实现MCP服务器

### 14.5.1 基础MCP服务器框架

让我们创建一个完整的MCP服务器实现：

```python
#!/usr/bin/env python3
"""
基础MCP服务器实现
"""

import json
import sys
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: callable


@dataclass
class Resource:
    uri: str
    name: str
    description: str
    mime_type: str
    content: Optional[bytes] = None


class MCPServer:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Resource] = {}
        self.prompts: Dict[str, Any] = {}
        self.initialized = False
    
    def register_tool(self, name: str, description: str, 
                     input_schema: Dict[str, Any], handler: callable):
        """注册工具"""
        self.tools[name] = Tool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler
        )
    
    def register_resource(self, uri: str, name: str, 
                        description: str, mime_type: str, 
                        content: Optional[bytes] = None):
        """注册资源"""
        self.resources[uri] = Resource(
            uri=uri,
            name=name,
            description=description,
            mime_type=mime_type,
            content=content
        )
    
    async def handle_request(self, request: Dict[str, Any]) -&gt; Dict[str, Any]:
        """处理MCP请求"""
        method = request.get("method")
        params = request.get("params", {})
        
        try:
            if method == "initialize":
                return await self._handle_initialize(params)
            elif method == "tools/list":
                return await self._handle_tools_list()
            elif method == "tools/call":
                return await self._handle_tools_call(params)
            elif method == "resources/list":
                return await self._handle_resources_list()
            elif method == "resources/read":
                return await self._handle_resources_read(params)
            elif method == "prompts/list":
                return await self._handle_prompts_list()
            elif method == "prompts/get":
                return await self._handle_prompts_get(params)
            else:
                return self._error_response(-32601, "Method not found")
        
        except Exception as e:
            return self._error_response(-32603, str(e))
    
    async def _handle_initialize(self, params: Dict[str, Any]) -&gt; Dict[str, Any]:
        """处理初始化请求"""
        self.initialized = True
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "serverInfo": {
                    "name": "demo-mcp-server",
                    "version": "1.0.0"
                }
            }
        }
    
    async def _handle_tools_list(self) -&gt; Dict[str, Any]:
        """处理列出工具请求"""
        tools_list = []
        for tool in self.tools.values():
            tools_list.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            })
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "tools": tools_list
            }
        }
    
    async def _handle_tools_call(self, params: Dict[str, Any]) -&gt; Dict[str, Any]:
        """处理工具调用请求"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            return self._error_response(-32602, f"Tool not found: {tool_name}")
        
        tool = self.tools[tool_name]
        
        try:
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)
            
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ]
                }
            }
        except Exception as e:
            return self._error_response(-32603, str(e))
    
    async def _handle_resources_list(self) -&gt; Dict[str, Any]:
        """处理列出资源请求"""
        resources_list = []
        for resource in self.resources.values():
            resources_list.append({
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mime_type
            })
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "resources": resources_list
            }
        }
    
    async def _handle_resources_read(self, params: Dict[str, Any]) -&gt; Dict[str, Any]:
        """处理读取资源请求"""
        uri = params.get("uri")
        
        if uri not in self.resources:
            return self._error_response(-32602, f"Resource not found: {uri}")
        
        resource = self.resources[uri]
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "contents": [
                    {
                        "uri": resource.uri,
                        "mimeType": resource.mime_type,
                        "text": resource.content.decode('utf-8') if resource.content else ""
                    }
                ]
            }
        }
    
    async def _handle_prompts_list(self) -&gt; Dict[str, Any]:
        """处理列出提示请求"""
        prompts_list = []
        for name, prompt in self.prompts.items():
            prompts_list.append({
                "name": name,
                "description": prompt.get("description", ""),
                "arguments": prompt.get("arguments", [])
            })
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "prompts": prompts_list
            }
        }
    
    async def _handle_prompts_get(self, params: Dict[str, Any]) -&gt; Dict[str, Any]:
        """处理获取提示请求"""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if name not in self.prompts:
            return self._error_response(-32602, f"Prompt not found: {name}")
        
        prompt = self.prompts[name]
        template = prompt.get("template", "")
        
        # 简单的模板替换
        for key, value in arguments.items():
            template = template.replace(f"{{{key}}}", str(value))
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": template
                        }
                    }
                ]
            }
        }
    
    def _error_response(self, code: int, message: str) -&gt; Dict[str, Any]:
        """创建错误响应"""
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            }
        }
    
    async def run_stdio(self):
        """使用stdio传输运行服务器"""
        print("MCP服务器启动，使用stdio传输", file=sys.stderr)
        
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                request = json.loads(line.strip())
                response = await self.handle_request(request)
                
                # 保留原始请求的id
                if "id" in request:
                    response["id"] = request["id"]
                
                print(json.dumps(response), flush=True)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                error_response = self._error_response(-32700, str(e))
                print(json.dumps(error_response), flush=True)


# 创建示例服务器
def create_demo_server():
    """创建演示用MCP服务器"""
    server = MCPServer()
    
    # 注册示例工具
    def calculator(expression: str) -&gt; str:
        """简单计算器"""
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    server.register_tool(
        name="calculator",
        description="执行数学计算",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式"
                }
            },
            "required": ["expression"]
        },
        handler=calculator
    )
    
    def get_weather(city: str) -&gt; str:
        """获取天气（模拟）"""
        weather_data = {
            "北京": "晴朗，25°C",
            "上海": "多云，22°C",
            "广州": "小雨，28°C",
            "深圳": "晴间多云，27°C"
        }
        return weather_data.get(city, f"未知城市: {city}")
    
    server.register_tool(
        name="get_weather",
        description="获取指定城市的天气信息",
        input_schema={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        },
        handler=get_weather
    )
    
    # 注册示例资源
    doc_content = """
    # MCP协议文档
    
    MCP（Model Context Protocol）是一个用于连接AI助手与外部数据源和工具的开放协议。
    
    主要特性：
    1. 标准化的工具调用接口
    2. 资源访问机制
    3. 提示模板系统
    4. 多种传输方式支持
    """.encode('utf-8')
    
    server.register_resource(
        uri="file:///docs/mcp-guide.md",
        name="MCP指南",
        description="MCP协议入门指南",
        mime_type="text/markdown",
        content=doc_content
    )
    
    # 注册示例提示
    server.prompts["code_review"] = {
        "description": "代码审查提示模板",
        "arguments": [
            {
                "name": "code",
                "description": "要审查的代码",
                "required": True
            }
        ],
        "template": """请审查以下代码，指出潜在的问题和改进建议：

代码：
{code}

请从以下方面进行审查：
1. 代码质量
2. 安全性
3. 性能
4. 可读性
"""
    }
    
    return server


if __name__ == "__main__":
    server = create_demo_server()
    asyncio.run(server.run_stdio())
```

### 14.5.2 运行MCP服务器

将上述代码保存为 `mcp_server.py`，然后运行：

```bash
python mcp_server.py
```

服务器将使用stdio传输方式等待客户端连接。

---

## 14.6 实现MCP客户端

### 14.6.1 基础MCP客户端框架

```python
#!/usr/bin/env python3
"""
基础MCP客户端实现
"""

import json
import sys
import asyncio
import subprocess
from typing import Dict, Any, List, Optional


class MCPClient:
    def __init__(self):
        self.request_id = 0
        self.transport = None
    
    async def connect_stdio(self, server_command: List[str]):
        """通过stdio连接到MCP服务器"""
        self.process = await asyncio.create_subprocess_exec(
            *server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.transport = "stdio"
        print(f"已连接到MCP服务器: {' '.join(server_command)}", file=sys.stderr)
    
    async def send_request(self, method: str, params: Dict[str, Any] = None) -&gt; Dict[str, Any]:
        """发送请求到服务器"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        if self.transport == "stdio":
            return await self._send_request_stdio(request)
        else:
            raise Exception("未连接到服务器")
    
    async def _send_request_stdio(self, request: Dict[str, Any]) -&gt; Dict[str, Any]:
        """通过stdio发送请求"""
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        # 读取响应
        response_line = await self.process.stdout.readline()
        if not response_line:
            raise Exception("服务器断开连接")
        
        return json.loads(response_line.strip())
    
    async def initialize(self) -&gt; Dict[str, Any]:
        """初始化连接"""
        return await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "demo-mcp-client",
                "version": "1.0.0"
            }
        })
    
    async def list_tools(self) -&gt; List[Dict[str, Any]]:
        """列出可用工具"""
        response = await self.send_request("tools/list")
        return response.get("result", {}).get("tools", [])
    
    async def call_tool(self, name: str, arguments: Dict[str, Any] = None) -&gt; Dict[str, Any]:
        """调用工具"""
        response = await self.send_request("tools/call", {
            "name": name,
            "arguments": arguments or {}
        })
        return response.get("result", {})
    
    async def list_resources(self) -&gt; List[Dict[str, Any]]:
        """列出可用资源"""
        response = await self.send_request("resources/list")
        return response.get("result", {}).get("resources", [])
    
    async def read_resource(self, uri: str) -&gt; Dict[str, Any]:
        """读取资源"""
        response = await self.send_request("resources/read", {
            "uri": uri
        })
        return response.get("result", {})
    
    async def list_prompts(self) -&gt; List[Dict[str, Any]]:
        """列出可用提示"""
        response = await self.send_request("prompts/list")
        return response.get("result", {}).get("prompts", [])
    
    async def get_prompt(self, name: str, arguments: Dict[str, Any] = None) -&gt; Dict[str, Any]:
        """获取提示"""
        response = await self.send_request("prompts/get", {
            "name": name,
            "arguments": arguments or {}
        })
        return response.get("result", {})
    
    async def close(self):
        """关闭连接"""
        if hasattr(self, 'process'):
            self.process.terminate()
            await self.process.wait()


# 使用示例
async def main():
    client = MCPClient()
    
    try:
        # 连接到服务器
        await client.connect_stdio(["python", "mcp_server.py"])
        
        # 初始化
        init_response = await client.initialize()
        print("初始化响应:", json.dumps(init_response, indent=2, ensure_ascii=False))
        
        # 列出工具
        tools = await client.list_tools()
        print("\n可用工具:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
        
        # 调用计算器工具
        result = await client.call_tool("calculator", {"expression": "25 * 4 + 10"})
        print("\n计算器结果:", result)
        
        # 调用天气工具
        weather_result = await client.call_tool("get_weather", {"city": "北京"})
        print("天气结果:", weather_result)
        
        # 列出资源
        resources = await client.list_resources()
        print("\n可用资源:")
        for resource in resources:
            print(f"  - {resource['uri']}: {resource['name']}")
        
        # 读取资源
        if resources:
            content = await client.read_resource(resources[0]['uri'])
            print("\n资源内容:")
            print(content)
        
        # 列出提示
        prompts = await client.list_prompts()
        print("\n可用提示:")
        for prompt in prompts:
            print(f"  - {prompt['name']}: {prompt['description']}")
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 14.7 在LangChain中使用MCP

### 14.7.1 LangChain MCP集成

让我们创建一个LangChain与MCP集成的示例：

```python
#!/usr/bin/env python3
"""
LangChain与MCP集成示例
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Type
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# 先导入我们之前创建的MCP客户端
from chapter14_mcp_client import MCPClient


class MCPToolWrapper(BaseTool):
    """MCP工具的LangChain包装器"""
    name: str
    description: str
    args_schema: Type[BaseModel]
    mcp_client: MCPClient
    tool_name: str
    
    def _run(self, **kwargs) -&gt; str:
        """同步运行（LangChain工具接口）"""
        # 注意：这里我们使用asyncio.run来调用异步方法
        # 在实际生产环境中，应该使用适当的异步集成
        result = asyncio.run(self._arun(**kwargs))
        return result
    
    async def _arun(self, **kwargs) -&gt; str:
        """异步运行"""
        try:
            response = await self.mcp_client.call_tool(self.tool_name, kwargs)
            content = response.get("content", [])
            if content:
                return content[0].get("text", "")
            return json.dumps(response, ensure_ascii=False)
        except Exception as e:
            return f"工具调用错误: {str(e)}"


def create_mcp_tools(mcp_client: MCPClient, tools_info: List[Dict]) -&gt; List[BaseTool]:
    """从MCP工具信息创建LangChain工具"""
    tools = []
    
    for tool_info in tools_info:
        name = tool_info["name"]
        description = tool_info["description"]
        input_schema = tool_info.get("inputSchema", {})
        
        # 动态创建参数模型
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        # 创建Pydantic模型
        fields = {}
        for prop_name, prop_info in properties.items():
            field_type = str
            if prop_info.get("type") == "integer":
                field_type = int
            elif prop_info.get("type") == "number":
                field_type = float
            elif prop_info.get("type") == "boolean":
                field_type = bool
            
            default = ... if prop_name in required else None
            fields[prop_name] = (field_type, Field(default=default, description=prop_info.get("description", "")))
        
        args_schema = type(f"{name.capitalize()}Args", (BaseModel,), fields)
        
        # 创建工具包装器
        tool = MCPToolWrapper(
            name=name,
            description=description,
            args_schema=args_schema,
            mcp_client=mcp_client,
            tool_name=name
        )
        tools.append(tool)
    
    return tools


async def langchain_mcp_example():
    """LangChain与MCP集成示例"""
    # 1. 创建并连接MCP客户端
    mcp_client = MCPClient()
    await mcp_client.connect_stdio(["python", "mcp_server.py"])
    await mcp_client.initialize()
    
    # 2. 获取MCP工具
    mcp_tools_info = await mcp_client.list_tools()
    print(f"从MCP服务器获取了 {len(mcp_tools_info)} 个工具")
    
    # 3. 转换为LangChain工具
    langchain_tools = create_mcp_tools(mcp_client, mcp_tools_info)
    print(f"创建了 {len(langchain_tools)} 个LangChain工具")
    
    # 4. 创建LLM（使用OpenAI兼容接口）
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        # 这里需要配置你的API密钥
        # api_key="your-api-key",
        # base_url="https://api.openai.com/v1"
    )
    
    # 5. 创建Agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手。使用提供的工具来回答问题。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, langchain_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=langchain_tools, verbose=True)
    
    # 6. 使用Agent
    print("\n=== 开始Agent交互 ===")
    
    # 测试查询1
    query1 = "帮我计算 15 * 6 + 30 等于多少？"
    print(f"\n用户: {query1}")
    result1 = await agent_executor.ainvoke({"input": query1})
    print(f"助手: {result1['output']}")
    
    # 测试查询2
    query2 = "北京今天的天气怎么样？"
    print(f"\n用户: {query2}")
    result2 = await agent_executor.ainvoke({"input": query2})
    print(f"助手: {result2['output']}")
    
    # 7. 清理
    await mcp_client.close()
    
    return agent_executor


# 资源读取示例
async def mcp_resource_example():
    """MCP资源读取示例"""
    mcp_client = MCPClient()
    await mcp_client.connect_stdio(["python", "mcp_server.py"])
    await mcp_client.initialize()
    
    # 列出资源
    resources = await mcp_client.list_resources()
    print(f"可用资源: {len(resources)}")
    
    # 读取资源
    if resources:
        resource_uri = resources[0]['uri']
        print(f"\n读取资源: {resource_uri}")
        content = await mcp_client.read_resource(resource_uri)
        
        print("\n资源内容:")
        for item in content.get('contents', []):
            print(item.get('text', ''))
    
    await mcp_client.close()


if __name__ == "__main__":
    print("=== LangChain MCP 集成示例 ===")
    print("\n注意：此示例需要配置OpenAI API密钥才能完整运行")
    print("你可以先测试资源读取功能\n")
    
    # 先运行资源示例（不需要LLM）
    asyncio.run(mcp_resource_example())
    
    # 如果配置了API密钥，可以运行完整示例
    # asyncio.run(langchain_mcp_example())
```

---

## 14.8 完整可运行示例

### 14.8.1 项目结构

让我们创建一个完整的可运行项目：

```
chapter14-mcp-example/
├── mcp_server.py          # MCP服务器
├── mcp_client.py          # MCP客户端
├── langchain_integration.py  # LangChain集成
├── requirements.txt       # 依赖
└── README.md             # 使用说明
```

### 14.8.2 requirements.txt

```txt
fastapi&gt;=0.104.0
uvicorn&gt;=0.24.0
langchain&gt;=0.1.0
langchain-openai&gt;=0.0.5
pydantic&gt;=2.0.0
python-dotenv&gt;=1.0.0
```

### 14.8.3 高级MCP服务器示例

让我们创建一个更高级的MCP服务器，包含文件系统访问功能：

```python
#!/usr/bin/env python3
"""
高级MCP服务器 - 文件系统访问
"""

import json
import sys
import asyncio
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path


# 导入基础服务器类（需要把之前的MCPServer放在同一目录）
# from mcp_server_base import MCPServer

# 这里我们直接继承之前的实现
class AdvancedMCPServer:
    def __init__(self, allowed_dir: str = None):
        self.allowed_dir = Path(allowed_dir) if allowed_dir else Path.cwd()
        self.tools: Dict[str, Any] = {}
        self.resources: Dict[str, Any] = {}
        self._register_tools()
        self._scan_resources()
    
    def _register_tools(self):
        """注册工具"""
        
        def list_files(path: str = ".") -&gt; str:
            """列出目录中的文件"""
            try:
                target_path = self.allowed_dir / path
                if not target_path.is_relative_to(self.allowed_dir):
                    return "错误：不允许访问该路径"
                
                files = []
                for item in target_path.iterdir():
                    file_type = "目录" if item.is_dir() else "文件"
                    files.append(f"{item.name} ({file_type})")
                
                return "\n".join(files) if files else "目录为空"
            except Exception as e:
                return f"错误: {str(e)}"
        
        self.tools["list_files"] = {
            "description": "列出指定目录中的文件",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "目录路径",
                        "default": "."
                    }
                }
            },
            "handler": list_files
        }
        
        def read_file(path: str) -&gt; str:
            """读取文件内容"""
            try:
                target_path = self.allowed_dir / path
                if not target_path.is_relative_to(self.allowed_dir):
                    return "错误：不允许访问该路径"
                
                with open(target_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) &gt; 10000:
                        content = content[:10000] + "\n\n[文件内容已截断...]"
                    return content
            except Exception as e:
                return f"错误: {str(e)}"
        
        self.tools["read_file"] = {
            "description": "读取文本文件内容",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    }
                },
                "required": ["path"]
            },
            "handler": read_file
        }
        
        def write_file(path: str, content: str) -&gt; str:
            """写入文件内容"""
            try:
                target_path = self.allowed_dir / path
                if not target_path.is_relative_to(self.allowed_dir):
                    return "错误：不允许访问该路径"
                
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"成功写入文件: {path}"
            except Exception as e:
                return f"错误: {str(e)}"
        
        self.tools["write_file"] = {
            "description": "写入文本文件内容",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "content": {
                        "type": "string",
                        "description": "文件内容"
                    }
                },
                "required": ["path", "content"]
            },
            "handler": write_file
        }
    
    def _scan_resources(self):
        """扫描资源"""
        # 这里可以添加资源发现逻辑
        pass
    
    async def handle_request(self, request: Dict[str, Any]) -&gt; Dict[str, Any]:
        """处理请求（简化版本）"""
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "advanced-file-server", "version": "1.0.0"}
                }
            }
        elif method == "tools/list":
            tools_list = []
            for name, tool in self.tools.items():
                tools_list.append({
                    "name": name,
                    "description": tool["description"],
                    "inputSchema": tool["inputSchema"]
                })
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {"tools": tools_list}
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name in self.tools:
                handler = self.tools[tool_name]["handler"]
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(**arguments)
                    else:
                        result = handler(**arguments)
                    
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [{"type": "text", "text": str(result)}]
                        }
                    }
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "error": {"code": -32603, "message": str(e)}
                    }
        
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {"code": -32601, "message": "Method not found"}
        }
    
    async def run_stdio(self):
        """运行stdio服务器"""
        print(f"高级MCP文件服务器启动，允许目录: {self.allowed_dir}", file=sys.stderr)
        
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                request = json.loads(line.strip())
                response = await self.handle_request(request)
                print(json.dumps(response), flush=True)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": str(e)}
                }), flush=True)


if __name__ == "__main__":
    # 使用当前目录作为允许的目录
    server = AdvancedMCPServer(allowed_dir=".")
    asyncio.run(server.run_stdio())
```

### 14.8.4 运行说明

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **启动MCP服务器**
```bash
python mcp_server.py
```

3. **运行客户端测试**
```bash
python mcp_client.py
```

4. **运行LangChain集成示例**
```bash
# 先配置OpenAI API密钥
export OPENAI_API_KEY="your-api-key"
python langchain_integration.py
```

---

## 14.9 最佳实践

### 14.9.1 安全最佳实践

1. **权限控制**
   - 实施最小权限原则
   - 对敏感操作进行用户授权
   - 记录所有操作日志

2. **输入验证**
   - 验证所有输入参数
   - 防止路径遍历攻击
   - 限制资源访问范围

3. **错误处理**
   - 不要在错误信息中暴露敏感信息
   - 提供友好的错误提示
   - 记录详细的错误日志

### 14.9.2 性能优化

1. **资源管理**
   - 实现资源缓存机制
   - 支持流式传输大文件
   - 合理设置超时时间

2. **并发处理**
   - 支持异步操作
   - 实现连接池
   - 限制并发请求数

### 14.9.3 开发建议

1. **模块化设计**
   - 分离传输层和业务逻辑
   - 使用插件系统扩展功能
   - 保持接口简洁

2. **文档完善**
   - 为工具提供清晰的描述
   - 包含使用示例
   - 记录版本变更

---

## ✅ 章节总结

### 核心要点回顾

1. **MCP协议定义**：MCP是连接AI助手与外部数据源和工具的开放协议，提供标准化接口
2. **三大核心能力**：工具（Tools）、资源（Resources）、提示模板（Prompts）
3. **架构模式**：客户端-服务器架构，支持stdio和HTTP等多种传输方式
4. **LangChain集成**：可以将MCP工具包装为LangChain工具，与Agent系统无缝集成
5. **安全考虑**：实施权限控制、输入验证和安全审计

### 关键术语

| 术语 | 解释 |
|------|------|
| MCP | Model Context Protocol，模型上下文协议 |
| JSON-RPC | 基于JSON的远程过程调用协议 |
| MCP服务器 | 提供工具、资源等功能的服务端组件 |
| MCP客户端 | 与AI助手集成的客户端组件 |
| 工具 | MCP提供的可执行函数 |
| 资源 | MCP提供的数据源 |
| 提示模板 | 预定义的提示词模板 |

### 下章预告

在下一章中，我们将学习Agent系统的评估与测试方法，包括如何评估Agent的性能、如何设计测试用例，以及如何持续优化Agent系统。

---

[← 返回课程目录](../course-overview.md)
