# -*- coding: utf-8 -*-
"""RAG 性能评估。

一、检索质量（无 LLM）：
  从知识库各文档抽取若干片段构造 query（ground truth = 该片段），
  统计 Recall@1/3/5 与 MRR@10。

二、生成质量（LLM-as-judge，本地 TRT-LLM Qwen 当裁判）：
  预置问答对 → RAG 回答 → 裁判对 faithfulness（忠实度）与 relevance（相关性）打 1-5 分。

用法：python scripts/evaluate_rag.py [--qa-only]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from company_rag import embedder, vector_store  # noqa: E402
from company_rag.config import settings  # noqa: E402
from company_rag.rag import query as rag_query  # noqa: E402

COLLECTION = settings.default_collection
RANDOM_SEED = 42


# ---------------- 一、检索质量 ----------------
def build_retrieval_queries(n_per_doc: int = 6) -> list[dict]:
    """从每份文档抽 chunk，用 chunk 首句+中段作 query，ground truth 为该 chunk。"""
    docs = vector_store.list_documents(COLLECTION)
    queries: list[dict] = []
    rng = random.Random(RANDOM_SEED)
    for d in docs:
        chunks = vector_store.get_chunks_by_doc(COLLECTION, d["doc_id"], limit=500)
        pool = [c for c in chunks if len(c["text"]) > 60]
        rng.shuffle(pool)
        for c in pool[:n_per_doc]:
            text = c["text"]
            # query = 首句（约 40-60 字），避免与片段完全一致导致作弊
            sentences = [s for s in text.replace("\n", " ").split("。") if s.strip()]
            query = (sentences[0][:60] + "？") if sentences else text[:60]
            queries.append({"query": query, "doc_id": c["metadata"].get("doc_id"), "chunk_id": c["id"]})
    return queries


def eval_retrieval(n_per_doc: int = 6, top_k_list=(1, 3, 5)) -> dict:
    queries = build_retrieval_queries(n_per_doc)
    hits_total = {k: 0 for k in top_k_list}
    mrr_sum = 0.0
    t0 = time.perf_counter()
    for i, q in enumerate(queries, 1):
        hits = vector_store.search(
            COLLECTION,
            embedder.embed_query(q["query"]),
            top_k=max(top_k_list),
        )
        rank = next((idx for idx, h in enumerate(hits) if h.id == q["chunk_id"]), None)
        for k in top_k_list:
            if rank is not None and rank < k:
                hits_total[k] += 1
        if rank is not None:
            mrr_sum += 1.0 / (rank + 1)
        if i % 20 == 0:
            print(f"  检索评估进度 {i}/{len(queries)}")
    n = len(queries)
    elapsed = time.perf_counter() - t0
    return {
        "queries": n,
        "recall@1": round(hits_total[1] / n, 4),
        "recall@3": round(hits_total[3] / n, 4),
        "recall@5": round(hits_total[5] / n, 4),
        "mrr@10": round(mrr_sum / n, 4),
        "elapsed_sec": round(elapsed, 1),
        "queries_per_sec": round(n / elapsed, 2),
    }


# ---------------- 二、生成质量（LLM-as-judge） ----------------
QA_PAIRS = [
    ("Transformer 论文提出的核心架构创新是什么？", ["attention_is_all_you_need"]),
    ("Transformer 中的自注意力机制如何工作？", ["attention_is_all_you_need"]),
    ("《数据安全法》规定数据处理者应履行哪些义务？", ["data_security_law"]),
    ("《数据安全法》何时通过、何时施行？", ["data_security_law"]),
    ("《个人信息保护法》中个人信息处理规则的核心要求？", ["personal_info_protection_law"]),
    ("处理敏感个人信息有什么特殊要求？", ["personal_info_protection_law"]),
    ("RAG 论文提出的方法由哪几个主要组件构成？", ["retrieval_augmented_generation"]),
    ("BERT 的预训练任务有哪些？", ["bert_pretraining_deep_bidirectional"]),
    ("Llama 2 论文的主要内容是什么？", ["llama2_open_foundation_models"]),
    ("BGE-M3 论文介绍了什么方法？", ["bge_m3_embedding"]),
]

JUDGE_PROMPT = (
    "你是严格的 RAG 质量裁判。给定【问题】【参考答案依据】【系统回答】，请评分：\n"
    "1. faithfulness 忠实度(1-5)：回答是否只依据给定资料、不编造（5=完全忠实，1=严重编造）\n"
    "2. relevance 相关性(1-5)：回答是否切题、有效回答问题（5=完全切题，1=答非所问）\n"
    "只输出 JSON：{{\"faithfulness\": 数字, \"relevance\": 数字}}\n\n"
    "【问题】{question}\n【资料依据】{context}\n【系统回答】{answer}\n"
)


async def eval_generation(llm=None, n: int = 10) -> dict:
    from company_rag.llm_client import LLMError, get_llm_client

    llm = llm or get_llm_client()
    pairs = QA_PAIRS[:n]
    scores = []
    detail = []
    for i, (question, hint) in enumerate(pairs, 1):
        resp = await rag_query(question, top_k=2)
        # judge 上下文要短：仅 1 条 snippet 前 100 字（judge 也受 448 token 预算限制）
        context = resp.sources[0].snippet[:100] if resp.sources else "（无检索结果）"
        judge_msgs = [
            {"role": "system", "content": "你是严格的评估裁判。"},
            {"role": "user", "content": JUDGE_PROMPT.format(question=question, context=context, answer=resp.answer)},
        ]
        try:
            judge = await llm.chat(judge_msgs, max_tokens=64, temperature=0.0)
            score = parse_score(judge.content)
        except LLMError as e:
            score = {"faithfulness": 0, "relevance": 0, "error": str(e)[:80]}
        scores.append(score)
        detail.append({"question": question, "answer": resp.answer[:200], "score": score, "truncated": resp.truncated})
        print(f"  [{i}/{len(pairs)}] {question[:30]}... -> {score}")
    ok = [s for s in scores if "error" not in s]
    return {
        "qa_pairs": len(pairs),
        "avg_faithfulness": round(sum(s.get("faithfulness", 0) for s in ok) / max(len(ok), 1), 2),
        "avg_relevance": round(sum(s.get("relevance", 0) for s in ok) / max(len(ok), 1), 2),
        "detail": detail,
    }


def parse_score(content: str) -> dict:
    import re

    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    f = re.search(r"faithfulness[\"':\s]+(\d)", content)
    r = re.search(r"relevance[\"':\s]+(\d)", content)
    return {"faithfulness": int(f.group(1)) if f else 0, "relevance": int(r.group(1)) if r else 0}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--qa-only", action="store_true")
    parser.add_argument("--n-per-doc", type=int, default=6)
    parser.add_argument("--qa-n", type=int, default=10)
    args = parser.parse_args()

    report: dict = {"collection": COLLECTION, "embed_model": settings.embed_model_path}
    if not args.qa_only:
        print("=== 一、检索质量评估 ===")
        report["retrieval"] = eval_retrieval(args.n_per_doc)
        print(json.dumps(report["retrieval"], ensure_ascii=False, indent=2))
    if not args.retrieval_only:
        print("\n=== 二、生成质量评估（LLM-as-judge）===")
        report["generation"] = await eval_generation(n=args.qa_n)
        print(json.dumps({k: v for k, v in report["generation"].items() if k != "detail"}, ensure_ascii=False, indent=2))

    out = Path(__file__).resolve().parents[1] / "results" / "rag_eval_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
