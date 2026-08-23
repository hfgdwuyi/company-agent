# -*- coding: utf-8 -*-
"""FastAPI 服务：对外 REST 接口。

端点一览：
  GET  /api/v1/health                      健康检查（LLM 连通性 + 知识库状态）
  POST /api/v1/documents/upload            上传文档（multipart，自动解析+入库）
  GET  /api/v1/documents                   文档清单
  GET  /api/v1/documents/{doc_id}          文档详情（前若干块）
  DELETE /api/v1/documents/{doc_id}        删除文档
  POST /api/v1/query                       RAG 问答
  POST /api/v1/query/stream                RAG 流式问答（SSE）
  POST /api/v1/chat                        Agent 对话（工具调用）
  POST /api/v1/chat/stream                 Agent 流式对话（SSE）
  GET  /api/v1/tools                       工具清单
  POST /api/v1/tools/invoke                直接调用工具（调试）
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import knowledge_base, vector_store
from . import __version__
from .agent import get_agent
from .config import settings
from .document_parser import SUPPORTED_EXTENSIONS
from .llm_client import LLMClient, LLMError, get_llm_client
from .models import (
    ChatRequest,
    ChatResponse,
    DocumentMeta,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SearchHitInfo,
    SearchRequest,
    SearchResponse,
    ToolCallRequest,
)
from .rag import query as rag_query
from .rag import retrieve as rag_retrieve
from .rag import stream_query
from .rag import _page_images as rag_page_images
from .tools import ToolExecutionError, get_registry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    logger.info("company_rag %s 启动，LLM=%s model=%s", __version__, settings.llm_base_url, settings.llm_model)
    yield
    await get_llm_client().close()


app = FastAPI(
    title="Company RAG & Agent Service",
    description="企业级 RAG 知识库 + Agent（文档解析 / 向量检索 / TensorRT-LLM 推理 / 工具调用）",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 鉴权（可选：设置 RAG_API_KEY 后启用） ----------
async def require_key(request: Request):
    if not settings.api_key:
        return
    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {settings.api_key}"
    if auth != expected:
        raise HTTPException(status_code=401, detail="Unauthorized: 缺少或错误的 API Key")


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("%s %s -> %d (%.0fms)", request.method, request.url.path, response.status_code, elapsed)
    return response


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ---------- 健康检查 ----------
@app.get("/api/v1/health", response_model=HealthResponse, dependencies=[Depends(require_key)])
async def health():
    llm_status: dict = {"reachable": False}
    try:
        models = await get_llm_client().list_models()
        llm_status = {"reachable": True, "models": models, "base_url": settings.llm_base_url}
    except Exception as e:  # noqa: BLE001
        llm_status = {"reachable": False, "error": str(e)[:200]}
    return HealthResponse(
        status="ok",
        llm=llm_status,
        kb={
            "collections": vector_store.collection_names(),
            "chunks": vector_store.count_chunks(settings.default_collection),
            "documents": len(vector_store.list_documents(settings.default_collection)),
        },
        version=__version__,
    )


# ---------- 文档管理 ----------
@app.post("/api/v1/documents/upload", response_model=DocumentMeta, dependencies=[Depends(require_key)])
async def upload_document(file: UploadFile = File(...), collection: str = settings.default_collection):
    ext = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型 {ext}，支持: {sorted(SUPPORTED_EXTENSIONS)}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    meta = knowledge_base.ingest_bytes(file.filename or "upload", data, collection=collection)
    if meta.status == "failed":
        raise HTTPException(status_code=422, detail=meta.error or "入库失败")
    return meta


@app.get("/api/v1/documents", dependencies=[Depends(require_key)])
async def list_documents(collection: str = settings.default_collection):
    return knowledge_base.list_documents(collection)


@app.get("/api/v1/documents/{doc_id}", dependencies=[Depends(require_key)])
async def get_document(doc_id: str, collection: str = settings.default_collection, limit: int = 20):
    chunks = knowledge_base.get_document_chunks(collection, doc_id, limit=limit)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"文档 {doc_id} 不存在")
    return {"doc_id": doc_id, "chunk_count_shown": len(chunks), "chunks": chunks}


@app.delete("/api/v1/documents/{doc_id}", dependencies=[Depends(require_key)])
async def delete_document(doc_id: str, collection: str = settings.default_collection):
    n = knowledge_base.delete_document(collection, doc_id)
    if n == 0:
        raise HTTPException(status_code=404, detail=f"文档 {doc_id} 不存在或已删除")
    return {"deleted": True, "doc_id": doc_id, "chunks_removed": n}


# ---------- RAG 问答 ----------
@app.post("/api/v1/query", response_model=QueryResponse, dependencies=[Depends(require_key)])
async def rag_qa(req: QueryRequest):
    try:
        resp = await _run_rag_query(req)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"LLM 服务错误: {e}") from e
    return resp


async def _run_rag_query(req: QueryRequest) -> QueryResponse:
    return await rag_query(
        req.question,
        collection=req.collection,
        top_k=req.top_k,
        doc_ids=req.doc_ids,
        llm=get_llm_client(),
    )


@app.post("/api/v1/query/stream", dependencies=[Depends(require_key)])
async def rag_qa_stream(req: QueryRequest):
    async def gen():
        async for evt in stream_query(
            req.question,
            collection=req.collection,
            top_k=req.top_k,
            doc_ids=req.doc_ids,
            llm=get_llm_client(),
        ):
            yield sse(evt)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------- Agent 对话 ----------
@app.post("/api/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_key)])
async def agent_chat(req: ChatRequest):
    try:
        resp = await get_agent().run(req.message, history=req.history, session_id=req.session_id)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"LLM 服务错误: {e}") from e
    return resp


@app.post("/api/v1/chat/stream", dependencies=[Depends(require_key)])
async def agent_chat_stream(req: ChatRequest):
    async def gen():
        async for evt in get_agent().run_stream(req.message, history=req.history, session_id=req.session_id):
            yield sse(evt)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------- 工具 ----------
@app.get("/api/v1/tools", dependencies=[Depends(require_key)])
async def list_tools():
    reg = get_registry()
    return {"tools": [t.to_openai_schema() for t in reg.tools.values()]}


@app.post("/api/v1/tools/invoke", dependencies=[Depends(require_key)])
async def invoke_tool(req: ToolCallRequest):
    reg = get_registry()
    try:
        result = reg.execute(req.name, req.arguments)
        return {"ok": True, "name": req.name, "arguments": req.arguments, "result": result}
    except ToolExecutionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# ---------- 文档检索（用户直接查询） ----------
@app.post("/api/v1/search", response_model=SearchResponse, dependencies=[Depends(require_key)])
async def search_documents(req: SearchRequest):
    """向量检索知识库：返回命中片段（不调用 LLM，快）。"""
    import asyncio

    hits = await asyncio.to_thread(
        rag_retrieve,
        req.query,
        collection=req.collection,
        top_k=req.top_k,
        doc_ids=req.doc_ids,
    )
    if req.min_score > 0:
        hits = [h for h in hits if h.score >= req.min_score]
    return SearchResponse(
        query=req.query,
        hits=[
            SearchHitInfo(
                doc_id=h.doc_id,
                filename=h.filename,
                page=h.page,
                chunk_index=h.chunk_index,
                score=round(h.score, 4),
                text=h.text,
                images=rag_page_images(h.doc_id, h.page),
            )
            for h in hits
        ],
        total=len(hits),
    )


# ---------- 图片服务 ----------
@app.get("/api/v1/images/{doc_id}/{filename}", dependencies=[Depends(require_key)])
async def serve_image(doc_id: str, filename: str):
    """提供 PDF 提取出的页面图片（data/images/{doc_id}/page_xxx/ 下）。"""
    from fastapi.responses import FileResponse

    # 防目录穿越
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    img_dir = settings.data_dir / "images" / doc_id
    for f in img_dir.rglob(filename):
        if f.is_file():
            media = "image/png" if f.suffix.lower() == ".png" else "image/jpeg"
            return FileResponse(f, media_type=media)
    raise HTTPException(status_code=404, detail=f"图片不存在: {doc_id}/{filename}")


# ---------- 用户查询页面 ----------
def _ui_candidates() -> list[Path]:
    """UI 页面查找顺序：环境变量指定目录（容器卷挂载）→ 项目根 web/ → 包内 web/。"""
    env_dir = os.environ.get("RAG_UI_DIR")
    cands = []
    if env_dir:
        cands.append(Path(env_dir) / "index.html")
    cands.append(Path(__file__).resolve().parents[2] / "web" / "index.html")
    cands.append(Path(__file__).parent / "web" / "index.html")
    return cands


@app.get("/ui", include_in_schema=False)
async def user_ui():
    from fastapi.responses import HTMLResponse

    for html in _ui_candidates():
        if html.exists():
            return HTMLResponse(html.read_text(encoding="utf-8"))
    raise HTTPException(status_code=500, detail="UI 页面文件缺失: web/index.html")


@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/ui")


@app.get("/index.html", include_in_schema=False)
async def index_html():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/ui")


@app.get("/api/v1/info", include_in_schema=False)
async def service_info():
    return {
        "service": "company-rag",
        "version": __version__,
        "docs": "/docs",
        "ui": "/ui",
        "endpoints": [
            "/api/v1/health",
            "/api/v1/documents/upload",
            "/api/v1/documents",
            "/api/v1/search",
            "/api/v1/query",
            "/api/v1/query/stream",
            "/api/v1/chat",
            "/api/v1/chat/stream",
            "/api/v1/tools",
        ],
    }
