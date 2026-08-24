# 构建 Qwen2.5-3B 微调模型（LoRA 已合并）INT4 TRT-LLM 引擎
# 在 trt-llm-qwen:v1 镜像内执行
set -e

MODEL_ROOT=/workspace/models_root_rag
SRC=$MODEL_ROOT/Qwen2.5-3B-MechRepair
CKPT=$MODEL_ROOT/ckpt_mech_3b_int4
ENGINE=$MODEL_ROOT/engine_mech_3b_int4

echo "== 1/2 量化 checkpoint (INT4) =="
python /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
  --model_dir $SRC \
  --output_dir $CKPT \
  --dtype float16 \
  --use_weight_only \
  --weight_only_precision int4 \
  --load_model_on_cpu

echo "== 2/2 构建 TRT 引擎 =="
trtllm-build \
  --checkpoint_dir $CKPT \
  --output_dir $ENGINE \
  --max_batch_size 8 \
  --max_input_len 4096 \
  --max_seq_len 4096 \
  --max_num_tokens 8192 \
  --max_beam_width 1 \
  --tokens_per_block 64 \
  --gemm_plugin float16 \
  --gpt_attention_plugin float16 \
  --context_fmha enable \
  --multiple_profiles enable \
  --remove_input_padding enable \
  --use_fused_mlp enable

echo "== 完成: $ENGINE =="
