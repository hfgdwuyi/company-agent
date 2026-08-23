# -*- coding: utf-8 -*-
"""RAG 管线：向量检索 → 上下文组装 → LLM 合成（带来源引用）。"""
from __future__ import annotations

import logging
from typing import Optional

from . import embedder, vector_store
from .config import settings
from .llm_client import LLMClient, get_llm_client
from .models import QueryResponse, SourceRef

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是企业知识库助手。只依据【参考资料】回答，不编造；资料不足时明确说明。"
    "回答规则：300 字以内，先给核心要点再补充细节，关键论断后标注来源编号如[1][2]。参考资料：\n"
)


def _build_context_prompt(hits: list[vector_store.SearchHit], budget_tokens: int) -> str:
    """按 token 预算组装上下文：从最相关块开始，装到预算为止。"""
    from .utils import estimate_tokens

    lines: list[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        page = f"，第{h.page}页" if h.page else ""
        piece = f"[{i}] 来源《{h.filename}》{page}\n{h.text}"
        piece_tokens = estimate_tokens(piece)
        if used + piece_tokens > budget_tokens and lines:
            break
        lines.append(piece)
        used += piece_tokens
    return "\n\n".join(lines)


def _page_images(doc_id: str, page: Optional[int]) -> list[str]:
    """返回某文档某页提取的图片 URL（data/images/{doc_id}/page_{n:03d}/ 下）。"""
    if page is None:
        return []
    img_dir = settings.data_dir / "images" / doc_id / f"page_{page:03d}"
    if not img_dir.is_dir():
        return []
    return [f"/api/v1/images/{doc_id}/{rel}" for rel in sorted(p.name for p in img_dir.iterdir() if p.is_file())]


def retrieve(
    question: str,
    collection: str = "",
    top_k: Optional[int] = None,
    doc_ids: Optional[list[str]] = None,
) -> list[vector_store.SearchHit]:
    """向量检索：返回按相似度降序的块。"""
    collection = collection or settings.default_collection
    top_k = top_k or settings.top_k
    qvec = embedder.embed_query(question)
    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    hits = vector_store.search(collection, qvec, top_k=top_k, where=where)
    if settings.score_threshold > 0:
        hits = [h for h in hits if h.score >= settings.score_threshold]
    return hits


async def query(
    question: str,
    collection: str = "",
    top_k: Optional[int] = None,
    doc_ids: Optional[list[str]] = None,
    llm: Optional[LLMClient] = None,
    max_tokens: Optional[int] = None,
) -> QueryResponse:
    """RAG 问答：检索 + 合成。"""
    collection = collection or settings.default_collection
    llm = llm or get_llm_client()
    hits = retrieve(question, collection=collection, top_k=top_k, doc_ids=doc_ids)
    sources = [
        SourceRef(
            doc_id=h.doc_id,
            filename=h.filename,
            page=h.page,
            chunk_index=h.chunk_index,
            score=round(h.score, 4),
            snippet=h.text[:200],
            images=_page_images(h.doc_id, h.page),
        )
        for h in hits
    ]
    if not hits:
        answer = "知识库中没有检索到与问题相关的资料，请换一种问法或先上传相关文档。"
        return QueryResponse(question=question, answer=answer, sources=[], usage={})

    context = _build_context_prompt(hits, budget_tokens=settings.llm_max_input_tokens - 180)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\n问题：{question}"},
    ]
    result = await llm.chat(messages, max_tokens=max_tokens, temperature=0.1)
    truncated = result.finish_reason == "length"
    if truncated:
        logger.warning("RAG 回答被截断 (finish_reason=length)，问题: %s", question[:60])
    return QueryResponse(
        question=question,
        answer=result.content.strip(),
        sources=sources,
        usage=result.usage,
        truncated=truncated,
    )


async def stream_query(
    question: str,
    collection: str = "",
    top_k: Optional[int] = None,
    doc_ids: Optional[list[str]] = None,
    llm: Optional[LLMClient] = None,
):
    """流式 RAG 问答：先检索并推送来源，再逐段推送回答。产出 dict 事件。"""
    collection = collection or settings.default_collection
    llm = llm or get_llm_client()
    hits = retrieve(question, collection=collection, top_k=top_k, doc_ids=doc_ids)
    sources = [
        SourceRef(
            doc_id=h.doc_id, filename=h.filename, page=h.page,
            chunk_index=h.chunk_index, score=round(h.score, 4), snippet=h.text[:200],
            images=_page_images(h.doc_id, h.page),
        )
        for h in hits
    ]
    if not hits:
        yield {"answer": "知识库中没有检索到相关资料。", "sources": [], "done": True}
        return
    context = _build_context_prompt(hits, budget_tokens=settings.llm_max_input_tokens - 180)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\n问题：{question}"},
    ]
    yield {"sources": [s.model_dump() for s in sources]}
    result = await llm.chat(messages, temperature=0.1, stream=True)
    text = result.content.strip()
    # 分片推送，模拟流式效果（TRT-LLM 全量返回后逐片下发）
    step = 16
    for i in range(0, len(text), step):
        yield {"delta": text[i : i + step]}
    yield {"done": True}
