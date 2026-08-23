# -*- coding: utf-8 -*-
"""嵌入模型：本地 bge-small-zh-v1.5（ModelScope 下载，hf-mirror 备选），CPU 运行不抢显存。"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

MODELSCOPE_MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODELSCOPE_CACHE = PROJECT_ROOT / "models"

_lock = threading.Lock()
_model = None
_model_path: Optional[str] = None


def _resolve_model_path() -> str:
    """配置路径 → ModelScope 缓存路径 → 触发下载。"""
    candidates = [
        Path(settings.embed_model_path),
        MODELSCOPE_CACHE / MODELSCOPE_MODEL_ID,
    ]
    for cand in candidates:
        if (cand / "config.json").exists():
            return str(cand)
    # 未找到 → 自动下载（ModelScope 优先，失败则 hf-mirror）
    logger.info("本地未找到嵌入模型 %s，尝试从 ModelScope 下载…", MODELSCOPE_MODEL_ID)
    try:
        from modelscope import snapshot_download

        p = snapshot_download(MODELSCOPE_MODEL_ID, cache_dir=str(MODELSCOPE_CACHE))
        if Path(p, "config.json").exists():
            return str(p)
    except Exception as e:  # noqa: BLE001
        logger.warning("ModelScope 下载失败: %s，改用 hf-mirror", e)
    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download as hf_download

    p = hf_download(MODELSCOPE_MODEL_ID, cache_dir=str(MODELSCOPE_CACHE))
    return p


def _get_model():
    """懒加载单例（线程安全）。"""
    global _model, _model_path
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            path = _resolve_model_path()
            logger.info("加载嵌入模型 %s (device=%s)", path, settings.embed_device)
            _model = SentenceTransformer(path, device=settings.embed_device)
            _model_path = path
    return _model


def embed_texts(texts: list[str], batch_size: Optional[int] = None) -> np.ndarray:
    """文档嵌入（不加检索指令），返回 L2 归一化向量 (n, dim)。"""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size or settings.embed_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_queries(queries: list[str]) -> np.ndarray:
    """查询嵌入：bge 系列查询需加指令前缀，检索效果更佳。"""
    if not queries:
        return np.zeros((0, 0), dtype=np.float32)
    instr = settings.embed_query_instruction
    prefixed = [f"{instr}{q}" if instr and not q.startswith(instr) else q for q in queries]
    return embed_texts(prefixed)


def embed_query(query: str) -> np.ndarray:
    return embed_queries([query])[0]


def embedding_dim() -> int:
    model = _get_model()
    try:
        return int(model.get_embedding_dimension())
    except AttributeError:  # sentence-transformers < 6.0
        return int(model.get_sentence_embedding_dimension())
