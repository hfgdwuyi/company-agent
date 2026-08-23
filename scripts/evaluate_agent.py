# -*- coding: utf-8 -*-
"""Agent 能力评估：预置任务集 → 断言工具选择/参数/回答质量。

用法：python scripts/evaluate_agent.py
输出：results/agent_eval_report.json（工具选择准确率、回答通过率、平均耗时等）
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from company_rag.agent import get_agent  # noqa: E402

# 任务集：prompt / 期望工具 / 期望参数断言（子串）/ 回答必须包含的关键词
TASKS = [
    {
        "name": "计算器-四则",
        "prompt": "帮我算一下 (128.5*3 + 75.25) / 4 等于多少？",
        "tool": "calculator",
        "arg_contains": ["128.5"],
        "answer_contains": ["115.1875", "115.19"],
    },
    {
        "name": "计算器-幂",
        "prompt": "2 的 10 次方是多少？",
        "tool": "calculator",
        "arg_contains": [],  # 参数表述多样（2**10 / 2^10 / pow(2,10) 均可），不强制
        "answer_contains": ["1024"],
    },
    {
        "name": "知识检索-个人信息保护法",
        "prompt": "用知识库查一下《个人信息保护法》中个人信息处理规则的核心要求是什么？",
        "tool": "retrieve_knowledge",
        "arg_contains": ["个人信息"],
        "answer_contains": ["同意", "合法", "目的"],
    },
    {
        "name": "知识检索-数据安全法",
        "prompt": "《数据安全法》对数据处理者有哪些要求？请查知识库回答。",
        "tool": "retrieve_knowledge",
        "arg_contains": ["数据安全法"],
        "answer_contains": ["安全", "数据"],
    },
    {
        "name": "文档清单",
        "prompt": "知识库里有哪些文档？",
        "tool": "list_documents",
        "arg_contains": [],
        "answer_contains": ["attention", "pdf"],
    },
    {
        "name": "当前时间",
        "prompt": "今天是几号？星期几？",
        "tool": "get_current_datetime",
        "arg_contains": [],
        "answer_contains": ["日", "星期"],
    },
    {
        "name": "文档数量统计",
        "prompt": "知识库里一共有几份文档？",
        "tool": "count_documents",
        "arg_contains": [],
        "answer_contains": ["12", "文档"],
    },
    {
        "name": "纯对话-无需工具",
        "prompt": "你好，简单介绍一下你自己。",
        "tool": None,  # 期望不调用工具
        "arg_contains": [],
        "answer_contains": ["助手", "知识库"],
    },
]


def check(prompt: str, tools, answer: str, task: dict) -> dict:
    """检查一次任务执行是否符合预期。"""
    expected_tool = task["tool"]
    tool_names = [t.name for t in tools]
    tool_ok = (expected_tool in tool_names) if expected_tool else (len(tool_names) == 0)

    args_ok = True
    if expected_tool:
        inv = next((t for t in tools if t.name == expected_tool), None)
        if inv is None:
            args_ok = False
        else:
            arg_str = json.dumps(inv.arguments, ensure_ascii=False)
            for frag in task.get("arg_contains", []):
                if frag.lower() not in arg_str.lower():
                    args_ok = False
                    break

    ans_low = answer.lower()
    answer_ok = any(k.lower() in ans_low for k in task.get("answer_contains", []))

    return {
        "tool_ok": tool_ok,
        "args_ok": args_ok,
        "answer_ok": answer_ok,
        "tools_used": tool_names,
        "answer": answer[:150],
    }


async def main() -> int:
    agent = get_agent()
    results = []
    t_all = time.perf_counter()
    for i, task in enumerate(TASKS, 1):
        t0 = time.perf_counter()
        resp = await agent.run(task["prompt"])
        elapsed = time.perf_counter() - t0
        r = check(task["prompt"], resp.tool_invocations, resp.answer, task)
        r["task"] = task["name"]
        r["prompt"] = task["prompt"]
        r["elapsed_sec"] = round(elapsed, 1)
        r["n_tools"] = len(resp.tool_invocations)
        results.append(r)
        ok = "✅" if (r["tool_ok"] and r["args_ok"] and r["answer_ok"]) else "❌"
        print(f"{ok} [{i}/{len(TASKS)}] {task['name']}: tools={r['tools_used']} "
              f"tool_ok={r['tool_ok']} args_ok={r['args_ok']} answer_ok={r['answer_ok']} ({elapsed:.1f}s)")

    total = len(results)
    passed = sum(1 for r in results if r["tool_ok"] and r["args_ok"] and r["answer_ok"])
    report = {
        "tasks": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4),
        "tool_select_accuracy": round(sum(1 for r in results if r["tool_ok"]) / total, 4),
        "answer_grounded_accuracy": round(sum(1 for r in results if r["answer_ok"]) / total, 4),
        "avg_elapsed_sec": round(sum(r["elapsed_sec"] for r in results) / total, 1),
        "avg_tools_per_task": round(sum(r["n_tools"] for r in results) / total, 2),
        "total_elapsed_sec": round(time.perf_counter() - t_all, 1),
        "detail": results,
    }
    print(f"\n=== Agent 评估报告 ===")
    print(f"通过率: {passed}/{total} ({report['pass_rate']:.1%})")
    print(f"工具选择准确率: {report['tool_select_accuracy']:.1%}")
    print(f"回答达标率: {report['answer_grounded_accuracy']:.1%}")
    print(f"平均耗时: {report['avg_elapsed_sec']}s / 任务, 平均工具调用 {report['avg_tools_per_task']} 次")

    out = Path(__file__).resolve().parents[1] / "results" / "agent_eval_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
