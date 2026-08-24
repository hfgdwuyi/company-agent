# -*- coding: utf-8 -*-
"""合并 LoRA 到基座 → 输出全量模型（供 TRT-LLM 构建引擎）。

用法：python scripts/merge_lora.py [--base-model 路径] [--lora 路径] [--out 路径]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"C:\AI\company-rag\models\Qwen\Qwen2.5-3B-Instruct")
LORA = Path(r"C:\AI\company-rag\data\lora\mech-repair-lora")
OUT = Path(r"C:\AI\company-rag\models\Qwen2.5-3B-MechRepair")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=str(BASE))
    parser.add_argument("--lora", default=str(LORA))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 关键：用 fp16 + CPU 加载（不用 4bit），保证 merge 输出干净 fp16 权重
    # （4bit base 上 merge_and_unload 在 transformers 5.x 不会真正反量化，会残留 uint8 量化权重）
    print("加载基座（fp16, CPU）...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print("加载并合并 LoRA（CPU，可能需要几分钟）...")
    model = PeftModel.from_pretrained(model, args.lora)
    merged = model.merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    print(f"合并模型已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
