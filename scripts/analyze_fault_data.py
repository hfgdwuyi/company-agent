# -*- coding: utf-8 -*-
"""分析维修文本的三段结构（故障现象/分析原因/解决方案）覆盖情况。"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(r"C:\AI\company-rag\data\fault_dataset\datasets--zhengr--Automotive_Industry_Fault_Data_Set\snapshots\bdcea0000664c43c74dc5a30076d1ba6604195ae\spo_0.json")

# 常见段落标记
P1 = re.compile(r"故障现象|现象|症状")
P2 = re.compile(r"故障原因|原因分析|分析原因|可能原因")
P3 = re.compile(r"处理方法|解决方案|解决措施|排除方法|诊断排除|处理措施|故障排除|维修方法|检修")


def split_sections(text: str) -> dict:
    """按标记切出 现象/原因/方案 三段（尽力而为）。"""
    out = {"现象": "", "原因": "", "方案": ""}
    # 找所有标记位置
    marks = []
    for name, pat in (("现象", P1), ("原因", P2), ("方案", P3)):
        for m in pat.finditer(text):
            marks.append((m.start(), name))
    marks.sort()
    if not marks:
        return out
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        seg = text[pos:end].strip()
        # 去掉标记词本身
        seg = re.sub(r"^(故障现象|现象|症状|故障原因|原因分析|分析原因|可能原因|处理方法|解决方案|解决措施|排除方法|诊断排除|处理措施|故障排除|维修方法|检修)[:：]?", "", seg).strip()
        if seg and len(seg) > 4:
            out[name] = seg
    return out


def main():
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    stats = {"总条数": len(rows), "三段齐全": 0, "只有现象": 0, "只有原因": 0, "只有方案": 0, "无标记": 0}
    samples = {"三段齐全": [], "无标记": []}
    for r in rows:
        sec = split_sections(r["input"])
        n = sum(1 for v in sec.values() if v)
        if n == 3:
            stats["三段齐全"] += 1
            if len(samples["三段齐全"]) < 2:
                samples["三段齐全"].append((r["input"][:100], sec))
        elif n == 0:
            stats["无标记"] += 1
            if len(samples["无标记"]) < 2:
                samples["无标记"].append(r["input"][:150])
        else:
            # 统计单段
            for k in ("现象", "原因", "方案"):
                if sec[k] and n == 1:
                    stats[f"只有{k}"] += 1
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    for tag, items in samples.items():
        print(f"\n=== {tag} 示例 ===")
        for src, sec in items:
            if isinstance(sec, dict):
                print("原文:", src[:80])
                for k, v in sec.items():
                    print(f"  {k}: {v[:80]}")
            else:
                print("原文:", src[:150])


if __name__ == "__main__":
    main()
