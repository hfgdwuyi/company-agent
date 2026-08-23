# -*- coding: utf-8 -*-
"""分块：中文友好的递归字符切分，保留文档/页码元数据。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from .config import settings
from .document_parser import ParsedDocument

logger = logging.getLogger(__name__)

# 中文优先的切分层级（粗 → 细）
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]


@dataclass
class Chunk:
    id: str
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """最细粒度：按字符硬切（仅当所有分隔符都失效时）。"""
    if len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def _recursive_split(text: str, chunk_size: int, overlap: int, seps: list[str]) -> list[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if not seps:
        return _hard_split(text, chunk_size, overlap)

    sep = seps[0]
    if sep:
        parts = [p for p in text.split(sep) if p]
        if len(parts) > 1:
            chunks: list[str] = []
            buf = ""
            for part in parts:
                if buf and len(buf) + len(sep) + len(part) > chunk_size:
                    chunks.append(buf)
                    buf = part
                else:
                    buf = (buf + sep + part) if buf else part
            if buf:
                chunks.append(buf)
            if len(chunks) > 1:
                return _apply_overlap(chunks, sep, overlap)
            # 只有一块（分隔符稀疏）→ 用更细分隔符重切
            return _recursive_split(buf, chunk_size, overlap, seps[1:])
    return _recursive_split(text, chunk_size, overlap, seps[1:])


def _apply_overlap(chunks: list[str], sep: str, overlap: int) -> list[str]:
    """相邻块重叠：后块头部补上前块尾部（保持上下文连贯）。"""
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out: list[str] = []
    for i, c in enumerate(chunks):
        if i == 0:
            out.append(c)
        else:
            prev_tail = chunks[i - 1][-overlap:]
            out.append(prev_tail + sep + c)
    return out


def chunk_document(doc: ParsedDocument, chunk_size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """把解析后的文档切成块。"""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    chunks: list[Chunk] = []
    idx = 0
    for page in doc.pages:
        for piece in _recursive_split(page.text, chunk_size, overlap, list(SEPARATORS)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    id=f"{doc.doc_id}::{idx}",
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    page=page.page_no,
                    chunk_index=idx,
                    text=piece,
                    metadata={
                        "source_type": doc.source_type,
                        "chunk_size": len(piece),
                    },
                )
            )
            idx += 1
    return chunks


def chunk_texts(texts: Iterable[str], chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """直接对文本列表分块（测试/工具用）。"""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    out: list[str] = []
    for t in texts:
        out.extend(_recursive_split(t, chunk_size, overlap, list(SEPARATORS)))
    return [c for c in (x.strip() for x in out) if c]
