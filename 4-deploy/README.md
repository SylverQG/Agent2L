# 部署运维指南

本目录包含 Agent 学习课程应用的部署与运维工具包，支持 Docker 和 Kubernetes 两种部署方式，并提供 Prometheus + Grafana 监控方案。

## 目录结构

```
4-deploy/
├── README.md                   # 本文件（部署说明）
├── kubernetes/                 # Kubernetes 部署清单（基于 Kustomize）
│   ├── README.md               # Kubernetes 详细部署说明
│   ├── kustomization.yaml      # Kustomize 配置入口
│   ├── namespace.yaml          # 命名空间定义
│   ├── deployment.yaml         # 应用部署配置
│   ├── service.yaml            # ClusterIP 服务
│   ├── ingress.yaml            # Ingress 入口（可选）
│   ├── secret.yaml.example     # 密钥配置示例
│   ├── pvc.yaml                # 持久化存储声明
│   └── hpa.yaml                # 自动扩缩容配置
├── docker/                     # Docker 部署方案
│   └── docker-compose.prod.yml # 生产环境 Docker Compose
└── monitoring/                 # 监控配置
    ├── prometheus.yml          # Prometheus 采集配置
    └── grafana-dashboard.json  # Grafana 监控面板
```

## 快速开始

### Docker 部署（本地/生产）

```bash
cp .env.example .env
docker compose -f 4-deploy/docker/docker-compose.prod.yml up -d
```

### Kubernetes 部署（集群）

```bash
kubectl apply -k 4-deploy/kubernetes/
```

### 监控

Prometheus 和 Grafana 可与应用一同部署。导入提供的 Grafana 面板即可监控关键指标。

## 组件说明

- **Kubernetes**：生产级容器编排，支持自动扩缩容、健康检查和持久化存储
- **Docker**：轻量级单节点部署，适合开发和中小规模生产
- **监控**：Prometheus 采集指标 + Grafana 可视化面板，提供运维洞察

## 环境变量

参考 `4-deploy/kubernetes/secret.yaml.example` 了解所需的密钥配置。生产环境建议使用专业的密钥管理方案（如 HashiCorp Vault、AWS Secrets Manager），切勿将密钥提交到版本控制中。