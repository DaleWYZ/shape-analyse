#!/bin/bash

# 构建Docker镜像
docker build -t shape-analyzer:latest .

# 应用Kubernetes配置
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml 