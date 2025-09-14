FROM python:3.10-slim

# 基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY start.sh ./start.sh
RUN chmod +x /app/start.sh

# 可选：模型权重放进容器（或运行时挂载）
# COPY weights/model.pt /app/weights/model.pt
# ENV MODEL_PATH=/app/weights/model.pt

EXPOSE 7860 8000
CMD ["/app/start.sh"]
