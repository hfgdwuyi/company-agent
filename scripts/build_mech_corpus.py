# -*- coding: utf-8 -*-
"""把 SFT 维修数据 + 操作规程组装成入库文档。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

sft_path = Path(r"C:\AI\company-rag\data\finetune\mech_repair_sft.jsonl")
out_path = Path(r"C:\AI\company-rag\data\finetune\raw_mech.txt")

sft = [json.loads(l) for l in open(sft_path, encoding="utf-8") if l.strip()]
parts = []
for r in sft:
    q = r["input"].replace("设备故障：", "").strip()
    parts.append("【维修案例】" + q + "\n" + r["output"])

guide = Path(r"C:\AI\company-rag\data\finetune\guides\production_equipment_safe_operation.txt")
if guide.exists():
    parts.append("【生产设备安全操作规程】\n" + guide.read_text(encoding="utf-8")[:6000])

out_path.write_text("\n\n".join(parts), encoding="utf-8")
print(f"语料已生成: {out_path} ({out_path.stat().st_size} 字节, {len(sft)} 案例)")
