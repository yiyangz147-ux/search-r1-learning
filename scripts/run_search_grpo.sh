#!/usr/bin/env bash
set -euo pipefail

# ====================== 可读、可修改的训练参数 ======================
# 数据固定：2000 train / 200 独立 dev / 原 1000 final test。
# 本轮授权：最多3 epochs；每题5条；只更新新增 rank32 q/v LoRA。
LR="${LR:-1e-6}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BATCH="${TRAIN_BATCH:-20}"
GROUP_SIZE=5
LORA_RANK=32
LORA_ALPHA=16
MICRO_BATCH=1
VALIDATE_EVERY=50
WAIT_GPU_SECONDS="${WAIT_GPU_SECONDS:-0}"
# GPU UUID 在 Python 入口固定为物理 2、3、4、5，不能用环境变量改到0/1。
# 512 token/action、2048 generated/question、3 searches、top3、6 turns
# 详见 configs/search_grpo_v1.yaml；最终实际值保存为 resolved_config.yaml。
# 分阶段运行时固定同一 RUN_ID，例如：
# RUN_ID=grpo-v1 bash scripts/run_search_grpo.sh preflight
# RUN_ID=grpo-v1 bash scripts/run_search_grpo.sh smoke
# RUN_ID=grpo-v1 bash scripts/run_search_grpo.sh train
# RUN_ID=grpo-v1 bash scripts/run_search_grpo.sh finish
# 一次完成全部阶段：RUN_ID=grpo-v1 bash scripts/run_search_grpo.sh all
# ==================================================================
ROOT=/srv/search-r1
PHASE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
RUN_ID="${RUN_ID:-grpo-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_STEPS=$((2000 / TRAIN_BATCH * EPOCHS))
export SEARCH_GRPO_ROOT="$ROOT"
export PYTHONPATH="$ROOT/scripts:$ROOT/vendor/verl-0.6.1"
export CUDA_VISIBLE_DEVICES=""  # parent/config inspection stays CPU-only
exec "$ROOT/.venv-grpo/bin/python" "$ROOT/scripts/run_search_grpo.py" \
  --phase "$PHASE" --run-id "$RUN_ID" --wait-gpu-seconds "$WAIT_GPU_SECONDS" \
  --override "actor_rollout_ref.actor.optim.lr=$LR" \
  --override "trainer.total_epochs=$EPOCHS" \
  --override "trainer.total_training_steps=$TRAIN_STEPS" \
  --override "actor_rollout_ref.actor.optim.total_training_steps=$TRAIN_STEPS" \
  --override "data.train_batch_size=$TRAIN_BATCH" \
  --override "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH" \
  --override "actor_rollout_ref.rollout.n=$GROUP_SIZE" \
  --override "actor_rollout_ref.model.lora_rank=$LORA_RANK" \
  --override "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA" \
  --override "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH" \
  --override "trainer.test_freq=$VALIDATE_EVERY" \
  --override "trainer.save_freq=$VALIDATE_EVERY" "$@"
