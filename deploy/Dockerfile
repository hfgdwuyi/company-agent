# company-rag 服务镜像
# 注意：嵌入模型与 Chroma 数据目录通过卷挂载（见 docker-compose.yml），不打包进镜像
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 系统依赖（chromadb 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && pip install --no-cache-dir uvicorn

COPY scripts ./scripts
COPY data ./data

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -fs http://127.0.0.1:8010/api/v1/health || exit 1

CMD ["uvicorn", "company_rag.api:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
