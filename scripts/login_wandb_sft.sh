#!/usr/bin/env bash
# Interactive authentication only. Never pass a key on the command line or
# redirect this command's input/output to a log.
set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
MARKER="$PROJECT_ROOT/.venv/.search-r1-lora-owned"
EXPECTED_MARKER="search-r1-lora-sft-venv-v1
$PROJECT_ROOT"

die() { printf '%s\n' "错误：$*" >&2; exit 1; }
[[ -t 0 && -t 1 ]] || die "请在服务器交互终端运行本脚本，并直接在 W&B 提示中填写 Key。不要通过聊天、命令参数或日志传递 Key。"
[[ -x "$VENV_PYTHON" && -f "$MARKER" && ! -L "$MARKER" ]] || die "请先运行 bash scripts/setup_lora_sft_env.sh。"
[[ "$(<"$MARKER")" == "$EXPECTED_MARKER" ]] || die "训练环境归属记录不匹配。"

# Preserve any existing WANDB_ENTITY and WANDB_PROJECT. WANDB_API_KEY, if
# already configured by the user, is left for W&B itself to handle.
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:30189}"
export HTTP_PROXY="${HTTP_PROXY:-$HTTPS_PROXY}"
export https_proxy="$HTTPS_PROXY"
export http_proxy="$HTTP_PROXY"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

printf '使用训练环境登录 W&B，并向服务端验证认证；已有认证会被复用。脚本不记录 Key。\n'
exec "$VENV_PYTHON" -m wandb login --verify
