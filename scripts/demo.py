# -*- coding: utf-8 -*-
"""控制台演示：RAG 问答 + Agent 工具调用（直连本服务模块，无需起 HTTP）。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from company_rag.agent import get_agent  # noqa: E402
from company_rag.rag import query as rag_query  # noqa: E402


def pprint_rag(q: str) -> None:
    print(f"\n{'='*70}\n📚 RAG 问答: {q}\n{'='*70}")
    resp = asyncio.run(rag_query(q))
    print(f"回答: {resp.answer}\n")
    if resp.sources:
        print("来源:")
        for s in resp.sources:
            print(f"  [{s.filename} 第{s.page or '?'}页 相似度{s.score:.3f}] {s.snippet[:60]}...")


def pprint_agent(q: str) -> None:
    print(f"\n{'='*70}\n🤖 Agent 对话: {q}\n{'='*70}")
    resp = asyncio.run(get_agent().run(q))
    if resp.tool_invocations:
        print("工具调用:")
        for t in resp.tool_invocations:
            mark = "✅" if t.ok else "❌"
            print(f"  {mark} {t.name}({t.arguments}) -> {t.result[:100]} ({t.elapsed_ms}ms)")
    print(f"回答: {resp.answer}")


if __name__ == "__main__":
    print("Demo: 输入问题（q=退出）")
    print("  前缀 'r:' = 纯 RAG 问答；其它 = Agent 对话")
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit"):
            break
        if q.startswith("r:"):
            pprint_rag(q[2:].strip())
        else:
            pprint_agent(q)
