#!/usr/bin/env bash
set +x
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
KEY_DIR="$PROJECT_ROOT/.secrets"
KEY_FILE="$KEY_DIR/deepseek_api_key"
if [[ ! -t 0 ]]; then
  printf '请在服务器交互终端运行本命令，Key 将隐藏输入。\n' >&2
  exit 1
fi
if [[ -e "$KEY_FILE" ]]; then
  printf 'Key 文件已存在，未覆盖：%s\n' "$KEY_FILE"
  exit 0
fi
umask 077
mkdir -p -- "$KEY_DIR"
chmod 700 -- "$KEY_DIR"
read -r -s -p 'DeepSeek API Key（输入隐藏）: ' TASK_DEEPSEEK_KEY
printf '\n'
if [[ -z "$TASK_DEEPSEEK_KEY" || "$TASK_DEEPSEEK_KEY" == *[[:space:]]* ]]; then
  unset TASK_DEEPSEEK_KEY
  printf 'Key 为空或含空白，未保存。\n' >&2
  exit 1
fi
set -o noclobber
printf '%s\n' "$TASK_DEEPSEEK_KEY" > "$KEY_FILE"
unset TASK_DEEPSEEK_KEY
chmod 600 -- "$KEY_FILE"
printf 'Key 已安全保存，未显示其内容：%s\n' "$KEY_FILE"
