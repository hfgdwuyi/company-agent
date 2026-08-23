# -*- coding: utf-8 -*-
"""向量库：ChromaDB 持久化（data/chroma），支持多集合、按文档删除、余弦相似度检索。"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import chromadb

from .config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_client: Optional[chromadb.ClientAPI] = None


def get_client() -> chromadb.ClientAPI:
    global _client
    with _lock:
        if _client is None:
            settings.ensure_dirs()
            _client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return _client


def _collection(name: str) -> Any:
    client = get_client()
    # 余弦空间：distance = 1 - cosine_similarity
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


@dataclass
class SearchHit:
    id: str
    doc_id: str
    filename: str
    page: Optional[int]
    chunk_index: int
    text: str
    score: float  # 余弦相似度 0~1
    metadata: dict = field(default_factory=dict)


def add_chunks(collection_name: str, ids: list[str], texts: list[str], embeddings: Any, metadatas: list[dict]) -> int:
    """批量写入块。embeddings: (n, dim) 已归一化。"""
    coll = _collection(collection_name)
    coll.add(ids=ids, documents=texts, embeddings=[e.tolist() for e in embeddings], metadatas=metadatas)
    return len(ids)


def search(
    collection_name: str,
    query_embedding: Any,
    top_k: int = 4,
    where: Optional[dict] = None,
) -> list[SearchHit]:
    """余弦检索，返回按相似度降序的命中。"""
    coll = _collection(collection_name)
    res = coll.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[SearchHit] = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for i, cid in enumerate(ids):
        m = metas[i] or {}
        score = max(0.0, min(1.0, 1.0 - float(dists[i])))
        hits.append(
            SearchHit(
                id=cid,
                doc_id=str(m.get("doc_id", "")),
                filename=str(m.get("filename", "")),
                page=int(m["page"]) if m.get("page") is not None else None,
                chunk_index=int(m.get("chunk_index", 0)),
                text=str(docs[i]),
                score=score,
                metadata=m,
            )
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def delete_document(collection_name: str, doc_id: str) -> int:
    """删除某文档的全部块。"""
    coll = _collection(collection_name)
    res = coll.get(where={"doc_id": doc_id}, include=[])
    ids = res.get("ids") or []
    if ids:
        coll.delete(ids=ids)
    return len(ids)


def list_documents(collection_name: str) -> list[dict]:
    """按 doc_id 聚合返回文档清单（文件名、页数、块数）。"""
    coll = _collection(collection_name)
    res = coll.get(include=["metadatas"])
    ids = res.get("ids") or []
    metas = res.get("metadatas") or []
    agg: dict[str, dict] = {}
    for cid, m in zip(ids, metas):
        m = m or {}
        doc_id = str(m.get("doc_id", ""))
        a = agg.setdefault(
            doc_id,
            {"doc_id": doc_id, "filename": str(m.get("filename", "")), "chunk_count": 0, "pages": set()},
        )
        a["chunk_count"] += 1
        if m.get("page") is not None:
            a["pages"].add(int(m["page"]))
    result = []
    for doc_id, a in agg.items():
        result.append(
            {
                "doc_id": doc_id,
                "filename": a["filename"],
                "chunk_count": a["chunk_count"],
                "pages": sorted(a["pages"]),
            }
        )
    result.sort(key=lambda d: d["filename"])
    return result


def count_chunks(collection_name: str) -> int:
    return _collection(collection_name).count()


def collection_names() -> list[str]:
    return [c.name for c in get_client().list_collections()]


def get_chunks_by_doc(collection_name: str, doc_id: str, limit: int = 500) -> list[dict]:
    """取某文档的块（工具/摘要用）。"""
    coll = _collection(collection_name)
    res = coll.get(where={"doc_id": doc_id}, include=["documents", "metadatas"], limit=limit)
    out = []
    for cid, doc, m in zip(res.get("ids") or [], res.get("documents") or [], res.get("metadatas") or []):
        out.append({"id": cid, "text": doc, "metadata": m or {}})
    out.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
    return out
