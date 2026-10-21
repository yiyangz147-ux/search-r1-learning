#!/usr/bin/env bash
# Run from any directory. No dependency installation or large-resource download.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$PROJECT_ROOT"

usage() {
  cat <<'USAGE'
用法：bash scripts/run_sft_teacher.sh [check|pilot|full|preview|export]

  check    检查输入、Key、模型列表、检索与学生 tokenizer；不生成或审查轨迹
  pilot    默认最多进行 20 次问题尝试，保留真实生成与审查结果（默认）
  full     在同一目录续跑，达到配额或候选耗尽时停止
  preview  将已有合格轨迹导出至独立预览目录，允许配额尚未完成
  export   配额完成后正式导出训练步骤

主要环境变量：
  SFT_PYTHON                 Python 可执行文件，默认 .venv/bin/python 或 python3
  DEEPSEEK_API_KEY            API Key，仅 check/pilot/full 需要
  DEEPSEEK_BASE_URL           默认 https://api.deepseek.com
  DEEPSEEK_MODEL              默认 deepseek-flash
  SEARCH_R1_RETRIEVER_URL     默认 http://127.0.0.1:8000/retrieve
  STUDENT_TOKENIZER          默认 Qwen/Qwen3-4B-Base，也可填本地 tokenizer 目录
  STUDENT_MAX_LENGTH         默认 4096
  SFT_RUN_NAME               默认 deepseek_v4_1_grounded_v3
  SFT_INPUT_DIR              默认 data/sft/selection_3000_clean
  SFT_OUTPUT_DIR             默认 data/sft/$SFT_RUN_NAME
  SFT_EXPORT_DIR             默认 ${SFT_OUTPUT_DIR}_sft
  SFT_PREVIEW_DIR            默认 ${SFT_OUTPUT_DIR}_preview
  SFT_CONCURRENCY            默认 2
  SFT_PILOT_LIMIT            默认 20，只允许正整数；设为 2 可先做极小试跑
  SFT_MAX_TOKENS             默认 4096（每轮教师输出预算，包含思考）
  SFT_REASONING_EFFORT       默认 low（教师）；显式 high 试验请使用新运行目录
  SFT_REVIEW_REASONING_EFFORT 默认 low（审查）
  SFT_PASSAGE_CHARS          默认 1800（每段展示给教师的字符上限）
  SFT_REVIEW_MODEL           默认与 DEEPSEEK_MODEL 相同
  SFT_ANSWER_ALIASES_FILE    可选：仅用于 SFT 训练题的有证据别名清单，原始严格 EM 不变

check 首次可能下载 tokenizer 文件，不下载 Qwen 模型权重。
pilot/full 会调用付费生成与审查 API；本脚本不会自动调用 full。
改动影响轨迹内容的配置后，请使用新的 SFT_RUN_NAME 或 SFT_OUTPUT_DIR。
预览/正式导出目录非空时拒绝覆盖，请设置新目录。
USAGE
}

MODE="${1:-pilot}"
if [[ "$MODE" == "-h" || "$MODE" == "--help" || "$MODE" == "help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -gt 1 ]]; then
  usage >&2
  exit 2
fi
case "$MODE" in
  check|pilot|full|preview|export) ;;
  *) usage >&2; exit 2 ;;
esac

if [[ -n "${SFT_PYTHON:-}" ]]; then
  PYTHON="$SFT_PYTHON"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON="python3"
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  printf '找不到 Python：%s。请建立虚拟环境或设置 SFT_PYTHON。\n' "$PYTHON" >&2
  exit 1
fi

RUN_NAME="${SFT_RUN_NAME:-deepseek_v4_1_grounded_v3}"
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  printf 'SFT_RUN_NAME 只能包含字母、数字、下划线、点和连字符，且以字母或数字开头。\n' >&2
  exit 2
fi
absolute_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$PROJECT_ROOT" "$1"
  fi
}
INPUT_DIR="$(absolute_path "${SFT_INPUT_DIR:-data/sft/selection_3000_clean}")"
OUTPUT_DIR="$(absolute_path "${SFT_OUTPUT_DIR:-data/sft/$RUN_NAME}")"
EXPORT_DIR="$(absolute_path "${SFT_EXPORT_DIR:-${OUTPUT_DIR}_sft}")"
PREVIEW_DIR="$(absolute_path "${SFT_PREVIEW_DIR:-${OUTPUT_DIR}_preview}")"
TOKENIZER="${STUDENT_TOKENIZER:-Qwen/Qwen3-4B-Base}"

printf '项目：%s\n模式：%s\nPython：%s\n输入：%s\n轨迹与审查：%s\n' \
  "$PROJECT_ROOT" "$MODE" "$PYTHON" "$INPUT_DIR" "$OUTPUT_DIR"

if [[ "$MODE" == "preview" || "$MODE" == "export" ]]; then
  DESTINATION="$EXPORT_DIR"
  if [[ "$MODE" == "preview" ]]; then
    DESTINATION="$PREVIEW_DIR"
  fi
  printf '导出：%s\n' "$DESTINATION"
  if [[ -e "$DESTINATION" ]]; then
    if [[ ! -d "$DESTINATION" ]] || [[ -n "$(find "$DESTINATION" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      printf '导出目标已存在且非空；拒绝覆盖。请设置新的 SFT_EXPORT_DIR 或 SFT_PREVIEW_DIR。\n' >&2
      exit 1
    fi
  fi
  COMMAND=("$PYTHON" "$PROJECT_ROOT/scripts/prepare_sft.py" export
    --input-dir "$OUTPUT_DIR" --output-dir "$DESTINATION")
  if [[ "$MODE" == "preview" ]]; then
    printf '只导出现有合格轨迹供检查；不会调用教师 API，也不表示已达到正式配额。\n'
    COMMAND+=(--allow-partial)
  else
    printf '正式导出要求所有质量门槛后的合格题数已达到配额。\n'
  fi
  exec "${COMMAND[@]}"
fi

printf '教师：%s\n检索：%s\n学生 tokenizer：%s\n' \
  "${DEEPSEEK_MODEL:-deepseek-flash}" \
  "${SEARCH_R1_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}" "$TOKENIZER"

COMMAND=("$PYTHON" "$PROJECT_ROOT/scripts/prepare_sft.py" generate
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --base-url "${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
  --model "${DEEPSEEK_MODEL:-deepseek-flash}"
  --retriever-url "${SEARCH_R1_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
  --generation-mode query-first
  --quality-review
  --student-tokenizer "$TOKENIZER"
  --student-max-length "${STUDENT_MAX_LENGTH:-4096}"
  --reasoning-effort "${SFT_REASONING_EFFORT:-low}"
  --max-tokens "${SFT_MAX_TOKENS:-4096}"
  --review-reasoning-effort "${SFT_REVIEW_REASONING_EFFORT:-low}"
  --review-max-tokens "${SFT_REVIEW_MAX_TOKENS:-4096}"
  --topk "${SFT_TOPK:-3}"
  --passage-chars "${SFT_PASSAGE_CHARS:-1800}"
  --min-searches "${SFT_MIN_SEARCHES:-1}"
  --max-searches "${SFT_MAX_SEARCHES:-3}"
  --max-attempts "${SFT_MAX_ATTEMPTS:-2}"
  --concurrency "${SFT_CONCURRENCY:-2}"
  --timeout "${SFT_TIMEOUT:-180}")
if [[ -n "${SFT_REVIEW_MODEL:-}" ]]; then
  COMMAND+=(--review-model "$SFT_REVIEW_MODEL")
fi
if [[ -n "${SFT_API_KEY_FILE:-}" ]]; then
  COMMAND+=(--api-key-file "$SFT_API_KEY_FILE")
fi
if [[ -n "${SFT_ANSWER_ALIASES_FILE:-}" ]]; then
  COMMAND+=(--answer-aliases-file "$SFT_ANSWER_ALIASES_FILE")
fi
case "$MODE" in
  check)
    printf '只检查连接、输入与 tokenizer，不生成或审查轨迹。首次可能下载 tokenizer 文件。\n'
    COMMAND+=(--preflight-only)
    ;;
  pilot)
    PILOT_LIMIT="${SFT_PILOT_LIMIT-20}"
    if [[ ! "$PILOT_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
      printf 'SFT_PILOT_LIMIT 必须是正整数，不能为 0、空值或其他文本；0 会表示无限续跑。\n' >&2
      exit 2
    fi
    printf '将调用真实 API：本次最多 %s 次问题尝试，成功与失败均保存，可稍后 full 续跑。\n' "$PILOT_LIMIT"
    COMMAND+=(--limit "$PILOT_LIMIT")
    ;;
  full)
    printf '将调用真实 API：按相同配置续跑至配额完成或候选耗尽，跳过已有合格题。\n'
    COMMAND+=(--limit 0)
    ;;
esac
exec "${COMMAND[@]}"
