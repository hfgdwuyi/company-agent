# -*- coding: utf-8 -*-
"""验证：RAG/检索返回带图片 URL + 图片端点可用。"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import httpx

base = "http://127.0.0.1:8010"

# 1. RAG 回答带图
r = httpx.post(base + "/api/v1/query", json={"question": "Transformer 模型的整体架构是什么样的？", "top_k": 3}, timeout=600).json()
print("answer:", r["answer"][:150])
for s in r["sources"]:
    print(f"source: {s['filename']} p{s['page']} images={s['images']}")

# 2. 图片端点
shown = False
for s in r["sources"]:
    for img_url in s.get("images", []):
        img = httpx.get(base + img_url, timeout=60)
        print(f"image endpoint {img_url}: {img.status_code} {img.headers.get('content-type')} {len(img.content)}B")
        shown = True
    if shown:
        break

# 3. 检索也带图
s = httpx.post(base + "/api/v1/search", json={"query": "Transformer 架构图", "top_k": 5}, timeout=120).json()
for h in s["hits"]:
    if h.get("images"):
        print("search hit with images:", h["filename"], "p" + str(h["page"]), h["images"])
