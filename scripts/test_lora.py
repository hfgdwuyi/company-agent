# -*- coding: utf-8 -*-
"""验证 LoRA 效果：加载 base + LoRA，测试工业标准格式回答。

用法：python scripts/test_lora.py [--base-model 路径] [--lora 路径]
对比：同一问题，基座模型 vs 微调模型。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"C:\AI\company-rag\models\Qwen\Qwen2.5-3B-Instruct")
LORA = Path(r"C:\AI\company-rag\data\lora\mech-repair-lora")

TEST_QUESTIONS = [
    "设备故障：减速器冬天启动困难，是什么原因？",
    "设备故障：液压泵运行时噪音很大，请分析原因并给出解决方案。",
    "设备故障：数控机床主轴突然停转，如何诊断和维修？",
    "设备故障：电机绕组发热冒烟，怎么处理？",
]

PROMPT = (
    "你是资深机械维修工程师。请按工业标准格式（问题现象/分析原因/解决方案）回答。\n"
    "### 指令：请按工业标准格式（问题现象/分析原因/解决方案）回答以下设备维修问题。\n"
    "### 问题：{q}\n"
    "### 回答："
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=str(BASE))
    parser.add_argument("--lora", default=str(LORA))
    parser.add_argument("--question", default=None, help="只测一个问题")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("=== 加载基座（4bit）+ LoRA ===")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    questions = [args.question] if args.question else TEST_QUESTIONS
    for q in questions:
        prompt = PROMPT.format(q=q)
        enc = tok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=400,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        answer = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print("=" * 60)
        print("Q:", q)
        print("A:", answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
