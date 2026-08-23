# -*- coding: utf-8 -*-
"""冒烟测试：分块 / 计算器 / 工具参数解析 / 向量库 CRUD。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from company_rag.chunker import chunk_texts  # noqa: E402
from company_rag.llm_client import parse_tool_arguments  # noqa: E402
from company_rag.tools import ToolExecutionError, get_registry  # noqa: E402


def test_chunker():
    text = ("第一条：员工考勤制度规定每日上下班需打卡。\n" * 30)
    chunks = chunk_texts([text], chunk_size=200, overlap=40)
    assert len(chunks) > 1, f"应切出多块, got {len(chunks)}"
    assert all(len(c) <= 240 for c in chunks), "块大小超限"
    print(f"✅ chunker: {len(chunks)} 块")


def test_calculator():
    reg = get_registry()
    assert reg.execute("calculator", {"expression": "(12.5*8+100)/4"}) == "50"
    assert reg.execute("calculator", {"expression": "2**10"}) == "1024"
    try:
        reg.execute("calculator", {"expression": "__import__('os')"})
        raise AssertionError("应拒绝危险表达式")
    except ToolExecutionError:
        pass
    print("✅ calculator: 安全计算通过")


def test_parse_tool_arguments():
    raw = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "北京"}}\n</tool_call>'
    args = parse_tool_arguments(raw)
    assert args["city"] == "北京", args
    assert parse_tool_arguments('{"a": 1}') == {"a": 1}
    print("✅ parse_tool_arguments: 兼容 <tool_call> 包裹")


def test_vector_store_crud():
    import numpy as np

    from company_rag import embedder, vector_store
    from company_rag.config import settings

    coll = "smoke_test"
    texts = ["华为在2024年研发投入达到1797亿元", "员工考勤制度要求每日打卡", "TensorRT-LLM 支持 int4 量化"]
    vecs = embedder.embed_texts(texts)
    ids = [f"t{i}" for i in range(3)]
    metas = [{"doc_id": "smoke", "filename": "t.txt", "page": 1, "chunk_index": i} for i in range(3)]
    vector_store.add_chunks(coll, ids, texts, vecs, metas)

    qv = embedder.embed_query("华为研发投入多少")
    hits = vector_store.search(coll, qv, top_k=2)
    assert hits and hits[0].chunk_index == 0, f"应命中研发投入块, got {hits}"
    assert vector_store.delete_document(coll, "smoke") == 3
    assert vector_store.count_chunks(coll) == 0
    print("✅ vector_store: 写入/检索/删除 通过")


if __name__ == "__main__":
    test_chunker()
    test_calculator()
    test_parse_tool_arguments()
    test_vector_store_crud()
    print("\n全部冒烟测试通过 🎉")
