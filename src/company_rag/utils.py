# -*- coding: utf-8 -*-
"""通用工具：token 估算（优先本地 Qwen tokenizer 精确计数，否则启发式兜底）。"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")

# 本地 Qwen2.5 tokenizer（与 TRT-LLM 引擎同源，无需下载）
TOKENIZER_CANDIDATES = [
    Path(r"C:\AI\TensorRT\Qwen2.5-7B-Instruct"),
    Path(__file__).resolve().parents[2] / "models" / "Qwen2.5-7B-Instruct",
]

_tokenizer = None
_tokenizer_lock = threading.Lock()


def _get_tokenizer():
    global _tokenizer
    with _tokenizer_lock:
        if _tokenizer is not None:
            return _tokenizer
        for cand in TOKENIZER_CANDIDATES:
            if (cand / "tokenizer.json").exists():
                try:
                    from transformers import AutoTokenizer

                    _tokenizer = AutoTokenizer.from_pretrained(str(cand), local_files_only=True)
                    logger.info("使用本地 tokenizer: %s", cand)
                    return _tokenizer
                except Exception as e:  # noqa: BLE001
                    logger.warning("加载 tokenizer 失败 %s: %s", cand, e)
        _tokenizer = False  # 标记不可用
    return None


def estimate_tokens(text: str) -> int:
    """估算文本 token 数。有本地 tokenizer 时精确计数，否则启发式。"""
    if not text:
        return 0
    tok = _get_tokenizer()
    if tok:
        try:
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:  # noqa: BLE001
            pass
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return max(1, int(cjk * 1.2 + other / 3) + 8)


def truncate_to_budget(text: str, budget_tokens: int) -> str:
    """按 token 预算截断文本（保留开头）。"""
    if estimate_tokens(text) <= budget_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= budget_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]
