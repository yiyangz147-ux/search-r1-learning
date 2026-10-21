#!/usr/bin/env bash
# Only explicit run/resume modes start training. All data preparation is on CPU.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
CONFIG="$PROJECT_ROOT/configs/lora_sft_train_v2.json"
DATA_CONFIG="$PROJECT_ROOT/configs/lora_sft_data_v2.json"
MODE="${1:-status}"

# ===== 训练参数：可直接改这里，也可在启动命令前设置同名环境变量 =====
export SFT_LORA_RANK="${SFT_LORA_RANK:-32}"
export SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-16}"
export SFT_LR="${SFT_LR:-0.00001}"
export SFT_EMBEDDING_LR="${SFT_EMBEDDING_LR:-0.00001}"
export SFT_EPOCHS="${SFT_EPOCHS:-4}"
export SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
export SFT_REWARD_EVERY="${SFT_REWARD_EVERY:-40}"
export SFT_REWARD_EXAMPLES="${SFT_REWARD_EXAMPLES:-32}"
# q_proj、v_proj 的 LoRA + 全部共享 embedding；其它 decoder 原始权重冻结。
# 每卡 batch=1，4卡全局 batch=4*1*8=32；完整轨迹最长4096。
# reward用独立开发集真实生成，total=Search-R1答案EM，format单独记录。
# GPU UUID 白名单固定为物理2、3、4、5，配置文件还会再次校验。


usage() {
  cat <<'USAGE'
用法：bash scripts/run_lora_sft_v2.sh [status|prepare|run|resume [latest|CHECKPOINT]]
  status   查看100条生成进度、混训数据和GPU/W&B配置；不启动模型
  prepare  仅CPU整理完整100条+Aether数据；未凑齐100条则退出
  run      整理v2数据，启动q/v LoRA + 全量共享embedding SFT
  resume   从同一配方的完整checkpoint续跑（含全量embedding），默认latest
首次在交互终端登录：bash scripts/login_wandb_sft.sh
USAGE
}
case "$MODE" in
  help|--help|-h) usage; exit 0 ;;
  status|prepare|run) [[ "$#" -le 1 ]] || { usage >&2; exit 2; } ;;
  resume) [[ "$#" -le 2 ]] || { usage >&2; exit 2; } ;;
  *) usage >&2; exit 2 ;;
esac
[[ -x "$PYTHON" ]] || { printf '先安装独立训练环境：bash scripts/setup_lora_sft_env.sh\n' >&2; exit 1; }
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export TMPDIR="$PROJECT_ROOT/.tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$PROJECT_ROOT/logs" "$TMPDIR" "$HF_HOME"

if [[ "$MODE" == status ]]; then
  exec "$PYTHON" "$PROJECT_ROOT/scripts/lora_launch_guard_v2.py" status
fi
if [[ "$MODE" == prepare ]]; then
  CUDA_VISIBLE_DEVICES='' exec "$PYTHON" "$PROJECT_ROOT/scripts/prepare_lora_sft_data_v2.py" --config "$DATA_CONFIG"
fi

export WANDB_MODE=online
export WANDB_DIR="$PROJECT_ROOT/logs/wandb"
export WANDB_LOG_MODEL=false WANDB_WATCH=false WANDB_DISABLE_CODE=true
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:30189}"
export HTTP_PROXY="${HTTP_PROXY:-$HTTPS_PROXY}"
export https_proxy="$HTTPS_PROXY" http_proxy="$HTTP_PROXY"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
mkdir -p "$WANDB_DIR"

# Check authentication before waiting or allocating model/GPU memory.
"$PYTHON" "$PROJECT_ROOT/scripts/lora_launch_guard_v2.py" precheck
STAMP="$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)"
RUN_LOG="$(mktemp "$PROJECT_ROOT/logs/lora-sft-v2-${MODE}-${STAMP}.XXXXXXXX.log")"
printf '日志：%s\n' "$RUN_LOG"
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$PROJECT_ROOT/scripts/prepare_lora_sft_data_v2.py" \
  --config "$DATA_CONFIG" --wait-for-generated --wait-timeout-seconds 86400 --poll-seconds 60 \
  2>&1 | tee -a "$RUN_LOG"

# Command substitution preserves the guard's exit status (process substitution would not).
GPU_VALUES="$("$PYTHON" "$PROJECT_ROOT/scripts/lora_launch_guard_v2.py" launch-values)"
mapfile -t GPU_LINES <<< "$GPU_VALUES"
[[ "${#GPU_LINES[@]}" == 2 && "${GPU_LINES[0]}" =~ ^[1-4]$ ]] || { printf 'GPU guard返回异常。\n' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${GPU_LINES[1]}" CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
# 固化本次sh参数。恢复时参数必须与原run完全一致，不能静默改变配方。
RESOLVED_CONFIG="$PROJECT_ROOT/configs/lora_sft_train_v2.resolved.json"
"$PYTHON" - "$CONFIG" "$RESOLVED_CONFIG" <<'PYCONFIG'
import json,os,sys
from pathlib import Path
config=json.loads(Path(sys.argv[1]).read_text())
for field,env,kind in [
    ("lora_rank","SFT_LORA_RANK",int),("lora_alpha","SFT_LORA_ALPHA",int),
    ("lr","SFT_LR",float),("embedding_lr","SFT_EMBEDDING_LR",float),
    ("epochs","SFT_EPOCHS",float),("gradient_accumulation","SFT_GRAD_ACCUM",int),
    ("reward_every_steps","SFT_REWARD_EVERY",int),("reward_examples","SFT_REWARD_EXAMPLES",int),
]: config[field]=kind(os.environ[env])
Path(sys.argv[2]).write_text(json.dumps(config,indent=2)+"\n")
print(json.dumps({"resolved_training_recipe":config},ensure_ascii=False),flush=True)
PYCONFIG
CONFIG="$RESOLVED_CONFIG"
COMMAND=("$PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="${GPU_LINES[0]}" "$PROJECT_ROOT/scripts/train_search_lora_sft_v2.py" --config "$CONFIG")
if [[ "$MODE" == resume ]]; then
  COMMAND+=(--resume-from-checkpoint "${2:-latest}")
fi
"${COMMAND[@]}" 2>&1 | tee -a "$RUN_LOG"
