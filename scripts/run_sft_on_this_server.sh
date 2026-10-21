#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export SFT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
export STUDENT_TOKENIZER=/srv/models/Qwen3-4B-Base
export HF_HOME=/srv/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SFT_RUN_NAME="${SFT_RUN_NAME:-deepseek_v4_1_grounded_100_metrics_v4}"
export SFT_INPUT_DIR="${SFT_INPUT_DIR:-data/sft/selection_100_clean}"
export SFT_ANSWER_ALIASES_FILE="${SFT_ANSWER_ALIASES_FILE:-$PROJECT_ROOT/data/sft/reviewed_answer_aliases.json}"
export SFT_REASONING_EFFORT="${SFT_REASONING_EFFORT:-low}"
export SEARCH_R1_RETRIEVER_URL="${SEARCH_R1_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
# This server cannot reach the official API directly. Reuse the user's existing
# SSH proxy, preserving HTTPS certificate verification and local retrieval.
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:30189}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:30189}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
MODE="${1:-pilot}"
if [[ "$MODE" != 'preview' && "$MODE" != 'export' && "$MODE" != '--help' && "$MODE" != 'help' && "$MODE" != '-h' ]]; then
  if [[ -z "${DEEPSEEK_API_KEY:-}" && -z "${SFT_API_KEY_FILE:-}" ]]; then
    if [[ -s "$PROJECT_ROOT/.secrets/deepseek_api_key" ]]; then
      export SFT_API_KEY_FILE="$PROJECT_ROOT/.secrets/deepseek_api_key"
    else
      printf '尚未配置 Key。请在服务器终端运行：\nbash %s/scripts/set_deepseek_key.sh\n' "$PROJECT_ROOT" >&2
      exit 1
    fi
  fi
fi
exec bash "$PROJECT_ROOT/scripts/run_sft_teacher.sh" "$@"
