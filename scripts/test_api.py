# -*- coding: utf-8 -*-
"""FastAPI 端点测试（真实服务 127.0.0.1:8010）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import httpx

BASE = "http://127.0.0.1:8010"
c = httpx.Client(timeout=600)


def show(title: str, r: httpx.Response, keys=None):
    print(f"\n=== {title} -> {r.status_code}")
    try:
        j = r.json()
        if keys:
            for k in keys:
                if k in j:
                    v = j[k]
                    print(f"  {k}: {str(v)[:400]}")
        else:
            print(json.dumps(j, ensure_ascii=False)[:600])
    except Exception:
        print(r.text[:300])


# 1. RAG 查询
show("POST /api/v1/query (RAG)", c.post(f"{BASE}/api/v1/query", json={"question": "Transformer 的核心创新是什么？", "top_k": 3}), ["answer", "sources"])
# 2. Agent 对话
show("POST /api/v1/chat (Agent 计算器)", c.post(f"{BASE}/api/v1/chat", json={"message": "帮我算 (256*0.15+88)/2"}), ["answer", "tool_invocations"])
# 3. Agent 对话（知识检索）
show("POST /api/v1/chat (Agent 检索)", c.post(f"{BASE}/api/v1/chat", json={"message": "知识库里的《数据安全法》对数据处理者有什么要求？"}), ["answer", "tool_invocations"])
# 4. 文档清单
show("GET /api/v1/documents", c.get(f"{BASE}/api/v1/documents"), None)
# 5. 工具清单
show("GET /api/v1/tools", c.get(f"{BASE}/api/v1/tools"), ["tools"])
# 6. 直接调用工具
show("POST /api/v1/tools/invoke", c.post(f"{BASE}/api/v1/tools/invoke", json={"name": "calculator", "arguments": {"expression": "2**10+5"}}), ["result"])
# 7. 上传新文档（用 data/raw 里的真实 PDF 测试上传接口）
import os

pdf = Path(r"C:\AI\company-rag\data\raw\lost_in_the_middle.pdf")
with pdf.open("rb") as f:
    show("POST /api/v1/documents/upload", c.post(f"{BASE}/api/v1/documents/upload", files={"file": (pdf.name, f, "application/pdf")}), ["doc_id", "chunk_count", "status"])
# 8. 删除刚上传的文档
doc_id = None
try:
    doc_id = json.loads(Path("last_upload.json").read_text(encoding="utf-8"))["doc_id"] if Path("last_upload.json").exists() else None
except Exception:
    pass
show("DELETE /api/v1/documents/uploaded", c.delete(f"{BASE}/api/v1/documents/__dummy__"))
# 9. 流式 RAG
print("\n=== POST /api/v1/query/stream (SSE) ===")
with c.stream("POST", f"{BASE}/api/v1/query/stream", json={"question": "RAG 论文的核心思想是什么？", "top_k": 2}) as r:
    print("status:", r.status_code)
    for line in r.iter_lines():
        if line.startswith("data:"):
            evt = json.loads(line[5:])
            if "delta" in evt:
                print(evt["delta"], end="", flush=True)
            elif "sources" in evt:
                print(f"\n[来源 {len(evt['sources'])} 条]")
            elif evt.get("done"):
                print("\n[stream done]")
print("\n\n全部 API 测试完成")
