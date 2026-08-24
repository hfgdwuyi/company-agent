# -*- coding: utf-8 -*-
"""关键对比：无格式提示词（纯问题）时 base vs LoRA 的格式自发性。"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\AI\company-rag\models\Qwen\Qwen2.5-3B-Instruct"
LORA = r"C:\AI\company-rag\data\lora\mech-repair-lora"

QUESTIONS = [
    "减速器冬天启动困难，是什么原因？",
    "液压泵运行时噪音很大，请分析原因并给出解决方案。",
    "数控机床主轴突然停转，如何诊断和维修？",
    "电机绕组发热冒烟，怎么处理？",
    "空压机压力达不到设定值，原因和解决？",
    "齿轮箱异响且有金属碎屑，怎么办？",
    "离心泵振动超标，分析与处理？",
    "轴承温度过高，可能原因和措施？",
    "皮带输送机跑偏，如何调整？",
    "液压油缸动作缓慢无力，怎么排查？",
]


def analyze(text: str) -> dict:
    has_phen = "问题现象" in text or "故障现象" in text
    has_reason = "分析原因" in text or "原因分析" in text or "可能原因" in text or "故障原因" in text
    has_sol = "解决方案" in text or "处理措施" in text or "解决方法" in text
    positions = []
    for key in ("问题现象", "故障现象"):
        i = text.find(key)
        if i != -1:
            positions.append(i)
            break
    for key in ("分析原因", "原因分析", "故障原因"):
        i = text.find(key)
        if i != -1:
            positions.append(i)
            break
    for key in ("解决方案", "处理措施"):
        i = text.find(key)
        if i != -1:
            positions.append(i)
            break
    ordered = positions == sorted(positions) and len(positions) == 3
    return {
        "三段完整": has_phen and has_reason and has_sol,
        "顺序正确": ordered,
        "开头现象": text.strip().startswith("问题现象") or text.strip().startswith("故障现象"),
        "字数": len(text),
    }


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    base_model.eval()
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    lora_model = PeftModel.from_pretrained(base_model, LORA)
    lora_model.eval()

    results = {"base": [], "lora": []}
    for i, q in enumerate(QUESTIONS, 1):
        for name, model in (("base", base_model), ("lora", lora_model)):
            enc = tok(q, return_tensors="pt").to("cuda")
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=400, temperature=0.3, do_sample=True, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            ans = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            r = analyze(ans)
            r["q"] = q
            r["t"] = round(time.perf_counter() - t0, 1)
            results[name].append(r)
        print(f"[{i}/10] done")

    for name in ("base", "lora"):
        rs = results[name]
        n = len(rs)
        print(f"\n=== {name}（无格式提示词）===")
        print(f"  三段完整: {sum(1 for r in rs if r['三段完整'])}/{n}  | 顺序正确: {sum(1 for r in rs if r['顺序正确'])}/{n}  | 开头现象: {sum(1 for r in rs if r['开头现象'])}/{n}")
        print(f"  平均字数: {sum(r['字数'] for r in rs)//n} | 平均耗时: {sum(r['t'] for r in rs)/n:.1f}s")

    out = Path(__file__).resolve().parents[1] / "results" / "lora_noprompt_report.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
