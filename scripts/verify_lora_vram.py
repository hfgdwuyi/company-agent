# -*- coding: utf-8 -*-
"""快速验证 7B QLoRA 在 8GB 卡的显存可行性（加载 + 一次前后向）。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch

BASE = Path(r"C:\AI\TensorRT\Qwen2.5-7B-Instruct")

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("显存总量:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,  # 省显存
)
print("加载 4bit 模型...")
model = AutoModelForCausalLM.from_pretrained(
    str(BASE),
    quantization_config=bnb,
    device_map="auto",
    max_memory={0: "6.3GB", "cpu": "48GB"},  # 强制部分层驻留 CPU，防 OOM
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained(str(BASE), trust_remote_code=True)
tok.pad_token = tok.eos_token
print(f"加载后显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

model = prepare_model_for_kbit_training(model)
lora = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
print(f"加 LoRA 后显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

# 一次前向+反向
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.config.use_cache = False
text = "你是资深机械维修工程师。\n### 指令：按工业标准格式回答\n### 问题：设备故障：减速器冬天启动困难\n### 回答：问题现象：减速器冬天启动困难"
enc = tok(text, return_tensors="pt", truncation=True, max_length=192).to("cuda")
out = model(**enc, labels=enc["input_ids"])
out.loss.backward()
print(f"一次训练步后显存: {torch.cuda.memory_allocated()/1e9:.2f}GB / 峰值: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
print("✅ 7B QLoRA 在 8GB 卡可行" if torch.cuda.max_memory_allocated() < 7.9e9 else "⚠️ 显存紧张，建议降级")
