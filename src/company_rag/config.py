# -*- coding: utf-8 -*-
"""全局配置：环境变量可覆盖（前缀 RAG_）。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"RAG_{name}", default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(f"RAG_{name}", str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(f"RAG_{name}", str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"RAG_{name}")
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # ---- 路径 ----
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(DEFAULT_DATA_DIR))))
    raw_dir: Path = field(default_factory=lambda: Path(_env("RAW_DIR", str(DEFAULT_DATA_DIR / "raw"))))
    parsed_dir: Path = field(default_factory=lambda: Path(_env("PARSED_DIR", str(DEFAULT_DATA_DIR / "parsed"))))
    chroma_dir: Path = field(default_factory=lambda: Path(_env("CHROMA_DIR", str(DEFAULT_DATA_DIR / "chroma"))))

    # ---- LLM（本地 Docker TensorRT-LLM，OpenAI 兼容） ----
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://127.0.0.1:8001/v1"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "engine_qwen_int4_v2"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 300))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1024))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.3))
    llm_max_concurrency: int = field(default_factory=lambda: _env_int("LLM_MAX_CONCURRENCY", 4))
    # 当前 TRT-LLM 引擎受显存限制，每请求最大输入约 448 tokens（KV cache 约束）
    llm_max_input_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_INPUT_TOKENS", 430))

    # ---- 嵌入模型（本地下载，ModelScope / hf-mirror） ----
    embed_model_path: str = field(
        default_factory=lambda: _env("EMBED_MODEL_PATH", str(DEFAULT_MODELS_DIR / "bge-small-zh-v1.5"))
    )
    embed_device: str = field(default_factory=lambda: _env("EMBED_DEVICE", "cpu"))  # cpu | cuda
    embed_batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 32))
    embed_query_instruction: str = field(
        default_factory=lambda: _env(
            "EMBED_QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关文章："
        )
    )

    # ---- 分块 ----
    # 受 TRT-LLM 448 token 输入限制，块宜小以便单次装入 2-3 块
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 220))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 50))

    # ---- 检索 ----
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 2))  # 输入预算有限，2 个块给输出留空间
    score_threshold: float = field(default_factory=lambda: _env_float("SCORE_THRESHOLD", 0.0))

    # ---- Agent ----
    agent_max_iterations: int = field(default_factory=lambda: _env_int("AGENT_MAX_ITERATIONS", 5))
    agent_max_tool_tokens: int = field(default_factory=lambda: _env_int("AGENT_MAX_TOOL_TOKENS", 256))

    # ---- 服务 ----
    api_host: str = field(default_factory=lambda: _env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8010))
    api_key: str = field(default_factory=lambda: _env("API_KEY", ""))  # 非空则启用 Bearer 鉴权

    # ---- 其它 ----
    default_collection: str = field(default_factory=lambda: _env("DEFAULT_COLLECTION", "enterprise_kb"))
    scan_pdf_ocr: bool = field(default_factory=lambda: _env_bool("SCAN_PDF_OCR", False))  # 扫描件 OCR（需装 pytesseract）

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.raw_dir, self.parsed_dir, self.chroma_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
