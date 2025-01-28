FROM python:3.8-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN groupadd -g 2000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash appuser

# 设置工作目录
WORKDIR /app

# 创建日志目录
RUN mkdir -p /app/logs && \
    chown -R appuser:appgroup /app/logs

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建必要的目录并设置权限
RUN mkdir templates uploads && \
    chown -R appuser:appgroup /app

# 复制应用代码和模板
COPY . .
COPY templates/* templates/

# 设置目录权限
RUN chown -R appuser:appgroup /app

# 切换到非 root 用户
USER appuser

# 暴露端口
EXPOSE 5000

# 启动应用
CMD ["python", "app.py"] 