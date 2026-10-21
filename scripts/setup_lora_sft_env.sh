#!/usr/bin/env bash
# Install dependencies only. This script does not import Torch, probe CUDA,
# log in to W&B, validate a model, or start training.
set -euo pipefail
umask 077

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
EXPECTED_ROOT="/srv/search-r1"
BASE_PYTHON="/usr/local/anaconda3/bin/python3.11"
WHEEL_DIR="/srv/data-preparation/search-r1/.qwen-eval-install/wheels"
VENV_DIR="$PROJECT_ROOT/.venv"
MARKER="$VENV_DIR/.search-r1-lora-owned"
MARKER_CONTENT="search-r1-lora-sft-venv-v1
$PROJECT_ROOT"

die() { printf '%s\n' "错误：$*" >&2; exit 1; }

[[ "$PROJECT_ROOT" == "$EXPECTED_ROOT" ]] || die "请先将训练包部署到 $EXPECTED_ROOT，再运行本脚本。"
[[ -x "$BASE_PYTHON" ]] || die "缺少基底 Python：$BASE_PYTHON"
[[ -d "$WHEEL_DIR" ]] || die "离线 wheel 缓存不存在：$WHEEL_DIR"
[[ -f "$PROJECT_ROOT/requirements-lora-sft.txt" ]] || die "训练依赖文件不存在。"
command -v flock >/dev/null 2>&1 || die "需要 Linux flock 命令，尚未创建或安装训练环境。"

# Keep every sizeable cache and temporary download off the server's root disk.
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export TMPDIR="$PROJECT_ROOT/.tmp"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
export PIP_REQUIRE_VIRTUALENV=true
unset PYTHONPATH PYTHONHOME PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
unset PIP_TARGET PIP_PREFIX PIP_USER PIP_LOG
mkdir -p "$PROJECT_ROOT/logs" "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME"

exec 9>"$PROJECT_ROOT/.setup-lora-sft.lock"
flock -n 9 || die "已有环境安装进程正在运行，请等待它结束。"

# Refuse an arbitrary pre-existing venv. A failed installation created here
# retains its marker and may be resumed by running this script again.
[[ ! -L "$VENV_DIR" ]] || die "拒绝使用符号链接形式的 .venv。"
if [[ -e "$VENV_DIR" ]]; then
    [[ -d "$VENV_DIR" && -f "$MARKER" && ! -L "$MARKER" ]] || die "已有 .venv 不属于本安装脚本；未修改它。"
    [[ "$(<"$MARKER")" == "$MARKER_CONTENT" ]] || die "已有 .venv 的归属记录不匹配；未修改它。"
else
    mkdir "$VENV_DIR"
    printf '%s\n' "$MARKER_CONTENT" > "$MARKER"
fi

INSTALL_LOG="$(mktemp "$PROJECT_ROOT/logs/install-lora-runtime.XXXXXXXX.log")"
on_exit() {
    local code=$?
    if [[ "$code" -ne 0 ]]; then
        printf '\n安装未完成。已有推理、检索和教师环境未被修改。\n日志：%s\n可修复网络或缓存后重新运行本脚本。\n' "$INSTALL_LOG" >&2
    fi
}
trap on_exit EXIT
run_logged() { "$@" 2>&1 | tee -a "$INSTALL_LOG"; }

if [[ ! -x "$VENV_DIR/bin/python" || ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
    run_logged "$BASE_PYTHON" -m venv "$VENV_DIR"
fi
VENV_PYTHON="$VENV_DIR/bin/python"
PIP_OPTIONS=(--disable-pip-version-check --no-input --progress-bar off --retries 2 --timeout 30)

printf '第一阶段：从现有 wheel 缓存安装固定的 Torch / Transformers 版本。\n'
if ! run_logged "$VENV_PYTHON" -m pip install "${PIP_OPTIONS[@]}" \
        --no-index --find-links "$WHEEL_DIR" \
        'torch==2.5.1+cu121' 'transformers==4.51.3' \
        'tokenizers==0.21.4' 'numpy==1.26.4'; then
    die "离线安装失败；请检查缓存中的包和依赖是否完整。没有改用其他环境。"
fi

# Existing proxy settings are retained; the server's known local forwarding
# proxy is the fallback. Do not echo proxy URLs or any credential variables.
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:30189}"
export HTTP_PROXY="${HTTP_PROXY:-$HTTPS_PROXY}"
export https_proxy="$HTTPS_PROXY"
export http_proxy="$HTTP_PROXY"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

printf '第二阶段：安装 LoRA、分布式训练、W&B 和数据依赖。需要保持现有代理隧道在线。\n'
run_logged "$VENV_PYTHON" -m pip install "${PIP_OPTIONS[@]}" \
    --find-links "$WHEEL_DIR" --index-url https://pypi.org/simple \
    -r "$PROJECT_ROOT/requirements-lora-sft.txt"

# The resolver installs/checks dependency compatibility; no model or GPU test
# is run here. Record the resolved runtime for the eventual training run.
"$VENV_PYTHON" -m pip freeze > "$PROJECT_ROOT/logs/runtime.freeze.txt.tmp"
mv "$PROJECT_ROOT/logs/runtime.freeze.txt.tmp" "$PROJECT_ROOT/logs/runtime.freeze.txt"
printf '\n训练依赖安装完成，尚未运行训练或测试。\n安装日志：%s\n版本记录：%s\n' \
    "$INSTALL_LOG" "$PROJECT_ROOT/logs/runtime.freeze.txt"
