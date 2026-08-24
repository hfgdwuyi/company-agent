# -*- coding: utf-8 -*-
"""对比评估：基座模型 vs LoRA 微调模型的工业标准格式遵循能力。

指标：问题现象段 / 分析原因段 / 解决方案段 完整率、顺序正确率、啰嗦率。
用法：python scripts/eval_lora_vs_base.py
"""
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
    "设备故障：减速器冬天启动困难，是什么原因？",
    "设备故障：液压泵运行时噪音很大，请分析原因并给出解决方案。",
    "设备故障：数控机床主轴突然停转，如何诊断和维修？",
    "设备故障：电机绕组发热冒烟，怎么处理？",
    "设备故障：空压机压力达不到设定值，原因和解决？",
    "设备故障：齿轮箱异响且有金属碎屑，怎么办？",
    "设备故障：离心泵振动超标，分析与处理？",
    "设备故障：轴承温度过高，可能原因和措施？",
    "设备故障：皮带输送机跑偏，如何调整？",
    "设备故障：液压油缸动作缓慢无力，怎么排查？",
]

PROMPT = (
    "你是资深机械维修工程师。请按工业标准格式（问题现象/分析原因/解决方案）回答。\n"
    "### 指令：请按工业标准格式（问题现象/分析原因/解决方案）回答以下设备维修问题。\n"
    "### 问题：{q}\n"
    "### 回答："
)


def analyze(text: str) -> dict:
    """检查回答是否包含标准三段、顺序、以及是否啰嗦。"""
    has_phen = "问题现象" in text or "故障现象" in text
    has_reason = "分析原因" in text or "原因分析" in text or "可能原因" in text or "故障原因" in text
    has_sol = "解决方案" in text or "处理措施" in text or "解决方法" in text
    # 顺序：现象 → 原因 → 方案
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
    verbose = len(text) > 700  # 超过 700 字视为啰嗦
    starts_with_phen = text.strip().startswith("问题现象") or text.strip().startswith("故障现象")
    return {
        "has_phen": has_phen, "has_reason": has_reason, "has_sol": has_sol,
        "ordered": ordered, "verbose": verbose, "starts_with_phen": starts_with_phen,
        "len": len(text),
    }


def generate(model, tok, prompt: str, device) -> str:
    import torch

    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=350, temperature=0.3, do_sample=True,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

    print("加载基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    base_model.eval()
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

    print("加载 LoRA 模型...")
    lora_model = PeftModel.from_pretrained(base_model, LORA)
    lora_model.eval()

    results = {"base": [], "lora": []}
    for i, q in enumerate(QUESTIONS, 1):
        prompt = PROMPT.format(q=q)
        for name, model in (("base", base_model), ("lora", lora_model)):
            t0 = time.perf_counter()
            ans = generate(model, tok, prompt, "cuda")
            r = analyze(ans)
            r["question"] = q
            r["elapsed"] = round(time.perf_counter() - t0, 1)
            results[name].append(r)
            print(f"[{i}/10] {name}: 三段={r['has_phen'] and r['has_reason'] and r['has_sol']} 顺序={r['ordered']} 开头现象={r['starts_with_phen']} 长度={r['len']} ({r['elapsed']}s)")
        if i % 3 == 0:
            print("  ...")

    def summary(name: str):
        rs = results[name]
        n = len(rs)
        return {
            "模型": name,
            "三段完整率": round(sum(1 for r in rs if r["has_phen"] and r["has_reason"] and r["has_sol"]) / n, 4),
            "顺序正确率": round(sum(1 for r in rs if r["ordered"]) / n, 4),
            "开头即现象率": round(sum(1 for r in rs if r["starts_with_phen"]) / n, 4),
            "啰嗦率": round(sum(1 for r in rs if r["verbose"]) / n, 4),
            "平均字数": round(sum(r["len"] for r in rs) / n, 0),
            "平均耗时(s)": round(sum(r["elapsed"] for r in rs) / n, 1),
        }

    print("\n=== 对比报告 ===")
    s_base, s_lora = summary("base"), summary("lora")
    print(json.dumps(s_base, ensure_ascii=False, indent=1))
    print(json.dumps(s_lora, ensure_ascii=False, indent=1))

    out = Path(__file__).resolve().parents[1] / "results" / "lora_vs_base_report.json"
    out.write_text(json.dumps({"base": s_base, "lora": s_lora, "detail": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
