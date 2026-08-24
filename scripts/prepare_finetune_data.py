# -*- coding: utf-8 -*-
"""构建机械维修 SFT 数据集（工业标准格式：问题现象/分析原因/解决方案）。

数据来源：
1. Automotive_Industry_Fault_Data_Set（1441 条真实维修文本）
   - 三段结构齐全 → 直接转换
   - 只有现象/原因/方案 → 用本地 TRT-LLM 合成补全为标准格式
2. 操作指南文本（另行补充，见 fetch 脚本）

输出：data/finetune/mech_repair_sft.jsonl
格式：{"instruction": "...", "input": "...", "output": "问题现象：...\n分析原因：...\n解决方案：..."}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import httpx

from company_rag.config import settings

FAULT_JSONL = Path(
    r"C:\AI\company-rag\data\fault_dataset\datasets--zhengr--Automotive_Industry_Fault_Data_Set\snapshots\bdcea0000664c43c74dc5a30076d1ba6604195ae\spo_0.json"
)
OUT = Path(r"C:\AI\company-rag\data\finetune\mech_repair_sft.jsonl")

P1 = re.compile(r"故障现象|现象|症状")
P2 = re.compile(r"故障原因|原因分析|分析原因|可能原因")
P3 = re.compile(r"处理方法|解决方案|解决措施|排除方法|诊断排除|处理措施|故障排除|维修方法|检修")

SYNTH_SYSTEM = (
    "你是资深机械维修工程师，拥有 20 年工业设备维修经验。"
    "请把给定的故障信息整理成标准维修报告，严格按以下格式输出，不要输出其它内容：\n"
    "问题现象：<一句话描述故障现象>\n"
    "分析原因：<2-4 条可能原因，每条一行，带序号>\n"
    "解决方案：<针对原因的对应处理措施，逐条对应>"
)

SYNTH_USER = "故障信息：{text}\n\n请按标准维修报告格式输出。"


def split_sections(text: str) -> dict:
    out = {"现象": "", "原因": "", "方案": ""}
    marks = []
    for name, pat in (("现象", P1), ("原因", P2), ("方案", P3)):
        for m in pat.finditer(text):
            marks.append((m.start(), name))
    marks.sort()
    if not marks:
        return out
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        seg = re.sub(
            r"^(故障现象|现象|症状|故障原因|原因分析|分析原因|可能原因|处理方法|解决方案|解决措施|排除方法|诊断排除|处理措施|故障排除|维修方法|检修)[:：]?",
            "",
            text[pos:end],
        ).strip()
        if seg and len(seg) > 2:
            out[name] = seg
    return out


def to_standard(sec: dict) -> str:
    return (
        f"问题现象：{sec['现象']}\n"
        f"分析原因：{sec['原因']}\n"
        f"解决方案：{sec['方案']}"
    )


def synth(text: str, client: httpx.Client) -> str:
    """用 TRT-LLM 把故障信息合成为标准格式。"""
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYNTH_SYSTEM},
            {"role": "user", "content": SYNTH_USER.format(text=text[:400])},
        ],
        "max_tokens": 300,
        "temperature": 0.3,
    }
    r = client.post(f"{settings.llm_base_url}/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", action="store_true", help="对不完整样本用 LLM 合成")
    parser.add_argument("--synth-limit", type=int, default=600, help="最多合成条数")
    parser.add_argument("--synth-batch", type=int, default=6, help="并发批大小")
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(FAULT_JSONL, encoding="utf-8") if l.strip()]
    direct: list[dict] = []
    need_synth: list[dict] = []
    for r in rows:
        sec = split_sections(r["input"])
        if (
            sec["现象"] and sec["原因"] and sec["方案"]
            and 4 <= len(sec["现象"]) <= 100  # 现象应是简短描述，过长多半切错
            and not sec["现象"].startswith(("；", "、", "1", "2", "3"))
        ):
            direct.append(
                {
                    "instruction": "请按工业标准格式（问题现象/分析原因/解决方案）回答以下设备维修问题。",
                    "input": f"设备故障：{sec['现象'][:120]}",
                    "output": to_standard(sec),
                    "source": "fault_dataset",
                }
            )
        else:
            # 取最完整的一段作线索
            clue = sec["现象"] or sec["原因"] or sec["方案"] or r["input"]
            need_synth.append({"clue": clue, "raw": r["input"]})

    print(f"规则直接可用: {len(direct)} 条 | 需合成: {len(need_synth)} 条")
    out_rows = list(direct)

    if args.synth:
        import concurrent.futures as cf

        limit = min(args.synth_limit, len(need_synth))
        todo = need_synth[:limit]
        client = httpx.Client(timeout=300)
        synth_system_tokens = len(SYNTH_SYSTEM)

        def work(item: dict) -> dict | None:
            try:
                text = item["clue"]
                if len(text) < 8:
                    text = item["raw"]
                out = synth(text, client)
                # 校验输出包含三段
                if "问题现象" in out and "分析原因" in out and "解决方案" in out:
                    return {
                        "instruction": "请按工业标准格式（问题现象/分析原因/解决方案）回答以下设备维修问题。",
                        "input": f"设备故障：{text[:120]}",
                        "output": out,
                        "source": "fault_dataset_synth",
                    }
            except Exception as e:  # noqa: BLE001
                print(f"  合成失败: {str(e)[:60]}")
            return None

        ok = 0
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=args.synth_batch) as ex:
            futures = [ex.submit(work, item) for item in todo]
            for i, f in enumerate(cf.as_completed(futures), 1):
                res = f.result()
                if res:
                    out_rows.append(res)
                    ok += 1
                if i % 50 == 0:
                    print(f"  合成进度 {i}/{len(todo)}，成功 {ok}，用时 {time.perf_counter()-t0:.0f}s")
        print(f"合成完成: {ok} 条 (用时 {time.perf_counter()-t0:.0f}s)")
        client.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"数据集已保存: {OUT}（共 {len(out_rows)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
