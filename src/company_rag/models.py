# -*- coding: utf-8 -*-
"""Pydantic 数据模型（API 对外 Schema）。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------- 文档 ----------
class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    source_type: str  # pdf | docx | txt | md
    size_bytes: int
    pages: int = 0
    chunk_count: int = 0
    status: str = "parsed"  # parsed | ingested | failed
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


# ---------- RAG 查询 ----------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="用户问题")
    collection: str = "enterprise_kb"
    top_k: int = Field(4, ge=1, le=50)
    doc_ids: Optional[list[str]] = None  # 限定文档范围
    stream: bool = False


class SourceRef(BaseModel):
    doc_id: str
    filename: str
    page: Optional[int] = None
    chunk_index: int = 0
    score: float = 0.0
    snippet: str = ""
    images: list[str] = Field(default_factory=list, description="该页提取的图片 URL 列表（/api/v1/images/...）")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    truncated: bool = Field(False, description="回答是否因输出长度限制被截断")


# ---------- Agent 对话 ----------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    history: list[ChatMessage] = Field(default_factory=list, description="历史消息（不含本条）")
    stream: bool = False


class ToolInvocation(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    ok: bool = True
    elapsed_ms: int = 0


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


# ---------- 文档检索（用户直接查询知识库） ----------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="检索关键词或问题")
    collection: str = "enterprise_kb"
    top_k: int = Field(10, ge=1, le=100)
    doc_ids: Optional[list[str]] = None  # 限定文档范围
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="最低相似度阈值")


class SearchHitInfo(BaseModel):
    doc_id: str
    filename: str
    page: Optional[int] = None
    chunk_index: int = 0
    score: float = 0.0
    text: str = ""
    images: list[str] = Field(default_factory=list, description="该页提取的图片 URL 列表")


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHitInfo] = Field(default_factory=list)
    total: int = 0


# ---------- 工具调试 ----------
class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


# ---------- 健康检查 ----------
class HealthResponse(BaseModel):
    status: str
    llm: dict[str, Any]
    kb: dict[str, Any]
    version: str
