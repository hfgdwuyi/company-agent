# -*- coding: utf-8 -*-
"""QLoRA 微调：Qwen2.5 系列 → 机械维修工业标准回答风格。

数据：data/finetune/mech_repair_sft.jsonl（instruction/input/output）
输出：data/lora/mech-repair-lora（PEFT adapter + tokenizer）
用法：python scripts/train_lora.py [--base-model 路径] [--epochs 3] [--rank 32] [--max-seq-len 512]
前提：8GB 显存紧张，请先停止 TRT-LLM 容器释放显存。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

BASE_MODEL = Path(r"C:\AI\company-rag\models\Qwen\Qwen2.5-3B-Instruct")
DATA = Path(r"C:\AI\company-rag\data\finetune\mech_repair_sft.jsonl")
OUT = Path(r"C:\AI\company-rag\data\lora\mech-repair-lora")

PROMPT_TEMPLATE = (
    "你是资深机械维修工程师。请按工业标准格式（问题现象/分析原因/解决方案）回答。\n"
    "### 指令：{instruction}\n"
    "### 问题：{input}\n"
    "### 回答：{output}"
)


def build_dataset(max_samples: int | None = None):
    """返回 (texts, prompt_lens)：prompt_lens 用于把 prompt 部分的 loss mask 掉（只学 output）。"""
    from datasets import Dataset

    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    if max_samples:
        rows = rows[:max_samples]
    texts, prompt_lens = [], []
    for r in rows:
        prompt = PROMPT_TEMPLATE.format(
            instruction=r["instruction"], input=r["input"], output=""
        )
        full = prompt + r["output"]
        texts.append(full)
        prompt_lens.append(len(prompt))
    print(f"数据集条数: {len(texts)}")
    return Dataset.from_dict({"text": texts, "prompt_len": prompt_lens})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=str(BASE_MODEL))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    base_path = Path(args.base_model)
    assert base_path.exists(), f"基座模型不存在: {base_path}"
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    print("=== 加载 4bit 量化基座 ===")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = build_dataset(args.max_samples)

    # tokenize：labels 中 prompt 部分设为 -100（只训练 output）
    def tokenize_fn(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
            return_attention_mask=False,
        )
        labels = []
        for i, ids in enumerate(enc["input_ids"]):
            prompt_tokens = len(
                tokenizer(examples["prompt_len"][i] and examples["text"][i][: examples["prompt_len"][i]] or "", add_special_tokens=False)["input_ids"]
            )
            lab = [-100] * prompt_tokens + ids[prompt_tokens:]
            labels.append(lab[: args.max_seq_len])
        return {"input_ids": enc["input_ids"], "labels": labels}

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    # 记录显存（观察是否 OOM）
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | 显存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB | 已用: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    training_args = TrainingArguments(
        output_dir=str(OUT / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        fp16=True,
        report_to=[],
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
    )

    print("=== 开始训练 ===")
    trainer.train()
    model.save_pretrained(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    print(f"LoRA 已保存: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
