FROM python:3.8-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建模板目录
RUN mkdir templates

# 复制应用代码和模板
COPY . .
COPY templates/* templates/

# 暴露端口
EXPOSE 5000

# 启动应用
CMD ["python", "app.py"] 