# -*- coding: utf-8 -*-
"""知识库服务：文档入库（解析→分块→嵌入→写入向量库）、删除、清单。"""
from __future__ import annotations

import hashlib
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from . import embedder, vector_store
from .chunker import chunk_document
from .config import settings
from .document_parser import ParsedDocument, parse_file
from .models import DocumentMeta

logger = logging.getLogger(__name__)


def make_doc_id(filename: str, size_bytes: int) -> str:
    stem = Path(filename).stem
    h = hashlib.md5(f"{filename}|{size_bytes}".encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{h}"


def ingest_path(
    path: Path,
    collection: Optional[str] = None,
    doc_id: Optional[str] = None,
    keep_copy: bool = True,
) -> DocumentMeta:
    """入库单个文件。返回文档元信息。doc_id 缺省按文件名+大小生成（幂等）。"""
    collection = collection or settings.default_collection
    path = Path(path)
    size = path.stat().st_size
    doc_id = doc_id or make_doc_id(path.name, size)

    t0 = time.perf_counter()
    doc = parse_file(path, doc_id=doc_id)
    if doc.error:
        return DocumentMeta(doc_id=doc_id, filename=path.name, source_type=doc.source_type, size_bytes=size, status="failed", error=doc.error)

    chunks = chunk_document(doc)
    if not chunks:
        return DocumentMeta(doc_id=doc_id, filename=path.name, source_type=doc.source_type, size_bytes=size, status="failed", error="解析后无有效文本（可能为扫描件）")

    # 幂等：先清旧块
    try:
        vector_store.delete_document(collection, doc_id)
    except Exception as e:  # noqa: BLE001
        logger.debug("删除旧块失败（首次入库正常）: %s", e)

    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": doc.filename,
            "page": c.page,
            "chunk_index": c.chunk_index,
            "source_type": c.metadata.get("source_type", ""),
        }
        for c in chunks
    ]
    vector_store.add_chunks(collection, [c.id for c in chunks], texts, embeddings, metadatas)

    if keep_copy:
        raw_dir = settings.raw_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / f"{doc_id}{path.suffix}"
        if not dest.exists():
            shutil.copy2(path, dest)

    elapsed = time.perf_counter() - t0
    logger.info("入库完成 %s: %d 块, %.1fs", path.name, len(chunks), elapsed)
    return DocumentMeta(
        doc_id=doc_id,
        filename=doc.filename,
        source_type=doc.source_type,
        size_bytes=size,
        pages=doc.page_count,
        chunk_count=len(chunks),
        status="ingested",
        metadata={**doc.metadata, "elapsed_sec": round(elapsed, 2), "collection": collection},
    )


def ingest_bytes(
    filename: str,
    data: bytes,
    collection: Optional[str] = None,
) -> DocumentMeta:
    """从字节入库（API 上传用）。doc_id 与 filename 元数据均基于原始文件名。"""
    collection = collection or settings.default_collection
    # 临时文件保留原始文件名（仅加短前缀防冲突），确保解析后 filename 元数据正确
    tmp_name = f"{uuid.uuid4().hex[:8]}_{Path(filename).name}"
    tmp = settings.parsed_dir / tmp_name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    doc_id = make_doc_id(filename, len(data))
    try:
        return ingest_path(tmp, collection=collection, doc_id=doc_id, keep_copy=True)
    finally:
        tmp.unlink(missing_ok=True)


def delete_document(collection: Optional[str], doc_id: str) -> int:
    collection = collection or settings.default_collection
    n = vector_store.delete_document(collection, doc_id)
    # 顺带清理 raw 副本
    for f in settings.raw_dir.glob(f"{doc_id}.*"):
        f.unlink(missing_ok=True)
    return n


def list_documents(collection: Optional[str] = None) -> list[dict]:
    collection = collection or settings.default_collection
    return vector_store.list_documents(collection)


def get_document_chunks(collection: Optional[str], doc_id: str, limit: int = 20) -> list[dict]:
    collection = collection or settings.default_collection
    return vector_store.get_chunks_by_doc(collection, doc_id, limit=limit)


def ingest_directory(directory: Path, collection: Optional[str] = None) -> list[DocumentMeta]:
    """批量入库目录下所有支持的文件。"""
    from .document_parser import SUPPORTED_EXTENSIONS

    results = []
    for p in sorted(Path(directory).iterdir()):
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                results.append(ingest_path(p, collection=collection))
            except Exception as e:  # noqa: BLE001
                logger.exception("批量入库失败 %s", p)
                results.append(
                    DocumentMeta(doc_id=p.stem, filename=p.name, source_type=p.suffix.lstrip("."), size_bytes=0, status="failed", error=str(e))
                )
    return results
