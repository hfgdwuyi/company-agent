# -*- coding: utf-8 -*-
"""端到端测试：RAG 问答 + Agent 工具调用（真实 TRT-LLM + 已入库真实语料）。"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from company_rag.agent import get_agent  # noqa: E402
from company_rag.rag import query as rag_query  # noqa: E402


def banner(t: str) -> None:
    print(f"\n{'='*72}\n{t}\n{'='*72}")


async def test_rag_english():
    banner("RAG 测试 1：英文论文（Attention Is All You Need）")
    t0 = time.perf_counter()
    resp = await rag_query("What is the core innovation of the Transformer architecture in the Attention Is All You Need paper?")
    print(f"回答({time.perf_counter()-t0:.1f}s): {resp.answer[:400]}")
    for s in resp.sources[:3]:
        print(f"  来源: {s.filename} 第{s.page}页 score={s.score:.3f}")


async def test_rag_chinese():
    banner("RAG 测试 2：中文法规（数据安全法）")
    t0 = time.perf_counter()
    resp = await rag_query("根据《数据安全法》，数据处理者开展数据处理活动应当遵循什么原则？")
    print(f"回答({time.perf_counter()-t0:.1f}s): {resp.answer[:400]}")
    for s in resp.sources[:3]:
        print(f"  来源: {s.filename} 第{s.page}页 score={s.score:.3f}")


async def test_rag_rag_paper():
    banner("RAG 测试 3：RAG 论文")
    t0 = time.perf_counter()
    resp = await rag_query("Retrieval-Augmented Generation for knowledge-intensive NLP tasks: what are the main components?")
    print(f"回答({time.perf_counter()-t0:.1f}s): {resp.answer[:400]}")
    for s in resp.sources[:2]:
        print(f"  来源: {s.filename} 第{s.page}页 score={s.score:.3f}")


async def test_agent_calculator():
    banner("Agent 测试 1：计算器工具")
    t0 = time.perf_counter()
    resp = await get_agent().run("帮我算一下 (128.5*3 + 75.25) / 4 等于多少？")
    print(f"耗时 {time.perf_counter()-t0:.1f}s")
    for t in resp.tool_invocations:
        print(f"  工具: {t.name}({t.arguments}) -> {t.result} ({t.elapsed_ms}ms)")
    print(f"回答: {resp.answer}")


async def test_agent_retrieve():
    banner("Agent 测试 2：知识检索工具")
    t0 = time.perf_counter()
    resp = await get_agent().run("用知识库查一下《个人信息保护法》中个人信息处理规则的核心要求是什么？")
    print(f"耗时 {time.perf_counter()-t0:.1f}s")
    for t in resp.tool_invocations:
        print(f"  工具: {t.name}({t.arguments}) -> {'OK' if t.ok else 'FAIL'} ({t.elapsed_ms}ms)")
    print(f"回答: {resp.answer[:500]}")


async def test_agent_list():
    banner("Agent 测试 3：文档清单工具")
    t0 = time.perf_counter()
    resp = await get_agent().run("知识库里有哪些文档？")
    print(f"耗时 {time.perf_counter()-t0:.1f}s")
    for t in resp.tool_invocations:
        print(f"  工具: {t.name}({t.arguments})")
    print(f"回答: {resp.answer[:400]}")


async def test_agent_datetime():
    banner("Agent 测试 4：时间工具")
    t0 = time.perf_counter()
    resp = await get_agent().run("今天是几号？星期几？")
    print(f"耗时 {time.perf_counter()-t0:.1f}s")
    for t in resp.tool_invocations:
        print(f"  工具: {t.name}({t.arguments}) -> {t.result}")
    print(f"回答: {resp.answer[:200]}")


async def main() -> None:
    await test_rag_english()
    await test_rag_chinese()
    await test_rag_rag_paper()
    await test_agent_calculator()
    await test_agent_retrieve()
    await test_agent_list()
    await test_agent_datetime()
    banner("端到端测试完成")


if __name__ == "__main__":
    asyncio.run(main())
