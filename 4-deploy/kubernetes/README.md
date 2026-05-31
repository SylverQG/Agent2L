# Kubernetes 部署指南

将 Agent 学习课程应用部署到 Kubernetes 集群的完整说明。

## 前置条件

- **kubectl** v1.28+ 已安装并配置
- 可访问的 Kubernetes 集群（minikube、kind 或云服务商）
- `kustomize`（kubectl v1.14+ 已内置）

## 快速部署

```bash
# 通过 Kustomize 应用所有资源
kubectl apply -k ./

# 验证部署状态
kubectl get all -n agent-course

# 查看 Pod 启动日志
kubectl get pods -n agent-course -w

# 本地端口转发访问应用
kubectl port-forward -n agent-course svc/agent-course 8888:8888
# 访问 http://localhost:8888
```

## 架构概览

```
                    ┌──────────────┐
                    │   Ingress    │  （可选）
                    │ agent-course │
                    │  .local      │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Service    │
                    │  ClusterIP   │
                    │   端口 8888  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Deployment  │
                    │ agent-course │
                    │  副本数: 1   │─── HPA（CPU > 70% 自动扩容）
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼─────┐ ┌───▼────┐ ┌────▼─────┐
       │   配置      │ │ 密钥   │ │   存储    │
       │ (环境变量)  │ │(外部)  │ │  10Gi    │
       └────────────┘ └────────┘ └──────────┘
```

## 组件说明

| 资源 | 说明 |
|------|------|
| Namespace | `agent-course` 命名空间，隔离所有资源 |
| Deployment | 运行 agent-course Jupyter 环境的主部署 |
| Service | ClusterIP 类型，对内暴露 8888 端口 |
| Ingress | （可选）对外路由，配置域名访问 |
| PVC | 持久化存储，保存课程数据和 Notebook |
| HPA | 自动扩缩容，CPU 利用率超 70% 触发 |
| Secret | API 密钥存储（需自行创建） |

## 部署步骤

### 1. 配置密钥

创建包含真实 API Key 的 Secret：

```bash
kubectl create secret generic agent-course-secret \
  --namespace agent-course \
  --from-literal=OPENAI_API_KEY='你的密钥' \
  --from-literal=ANTHROPIC_API_KEY='你的密钥'
```

### 2. 应用清单

```bash
kubectl apply -k ./
```

### 3. 验证部署

```bash
kubectl -n agent-course get pods
kubectl -n agent-course get svc
kubectl -n agent-course get hpa
```

### 4. 访问应用

```bash
# 端口转发本地访问
kubectl port-forward -n agent-course svc/agent-course 8888:8888

# 浏览器打开 http://localhost:8888
```

## 清理资源

```bash
kubectl delete -k ./
```