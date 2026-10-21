#!/usr/bin/env python3
"""Export compact predictions and a Chinese summary after all 2000 terminate.

CPU-only postprocessing. This file is intentionally independent of the running
evaluation's bound CODE_FILES and never changes its scripts or configuration.
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from prepare_grpo_lora_eval import (canonical, fail, load_config, load_grpo,
                                    read_json, selected_hash)
from report_grpo_lora_eval import report


ERROR_NAMES = {
    "format_failure": "格式解析失败",
    "unfinished_generation": "生成未完成",
    "search_budget_error": "搜索预算超限",
    "context_budget_error": "上下文预算超限",
    "token_budget_error": "累计生成 token 预算超限",
    "retrieval_failure": "检索服务失败",
    "model_failure": "模型失败",
    "oom": "显存不足 OOM",
}


def atomic_text(path, content):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require_complete(result):
    overall = result.get("overall", {})
    if (result.get("complete") is not True or overall.get("complete") is not True
            or overall.get("selected") != 2000 or overall.get("attempted") != 2000
            or overall.get("pending") != 0 or result.get("accuracy_final") is None):
        fail("Only a completed 2000-question report with pending=0 can be exported")
    for source in ("nq", "hotpotqa"):
        values = result.get("by_source", {}).get(source, {})
        if (values.get("complete") is not True or values.get("selected") != 1000
                or values.get("attempted") != 1000 or values.get("pending") != 0):
            fail(f"The {source} source report is not a completed 1000-question report")


def compact_predictions(config, result, examples):
    """Stream the already-validated workers; retain only each ID's final state."""
    output = Path(config["output_dir"])
    manifest = read_json(output / "run_manifest.json")
    if manifest.get("selected_sha256") != selected_hash(examples):
        fail("Original evaluation records changed after final reporting")
    expected = {row["sample_id"]: (index, row) for index, row in enumerate(examples)}
    final = {}
    count = manifest["num_workers"]
    for worker in range(count):
        path = output / f"worker-{worker}.predictions.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Match the reporter's treatment of an unfinished final append.
                    if not line.endswith("\n"):
                        break
                    raise
                sid = row.get("sample_id")
                if sid not in expected:
                    fail("A worker changed after reporting and contains an unknown sample ID")
                index, source = expected[sid]
                if (index % count != worker or row.get("worker_index") != worker
                        or row.get("source_index") != index):
                    fail(f"Worker ownership changed after reporting: {sid}")
                if any(row.get(key) != source[key] for key in ("question", "data_source", "golden_answers")):
                    fail(f"Prediction source changed after reporting: {sid}")
                final[sid] = {
                    "sample_id": sid, "source": row["data_source"], "question": row["question"],
                    "answer": row["answer"], "golden_answers": row["golden_answers"],
                    "status": row["status"], "em": int(row["em"]), "f1": row["f1"],
                    "search_count": row["search_count"],
                    "generated_tokens": row.get("usage", {}).get("generated_tokens", row.get("generated_tokens", 0)),
                    "error": row.get("error") or row.get("error_type"),
                }
    if len(final) != 2000 or set(final) != set(expected):
        fail("Worker terminal predictions no longer contain exactly the expected 2000 IDs")
    ordered = [final[row["sample_id"]] for row in examples]
    for source in (None, "nq", "hotpotqa"):
        values = ordered if source is None else [row for row in ordered if row["source"] == source]
        scores = result["overall"] if source is None else result["by_source"][source]
        if (len(values) != scores["selected"] or sum(row["em"] for row in values) != scores["correct"]
                or dict(Counter(row["status"] for row in values)) != scores["status_counts"]
                or sum(row["generated_tokens"] for row in values) != scores["total_generated_tokens"]
                or not math.isclose(sum(row["f1"] for row in values) / len(values), scores["f1"], abs_tol=1e-12)
                or not math.isclose(sum(row["search_count"] for row in values) / len(values), scores["mean_search_count"], abs_tol=1e-12)):
            fail("Workers changed while exporting; regenerate the final report before export")
    return ordered


def summary_markdown(config, result):
    overall = result["overall"]
    output = Path(config["output_dir"])
    valid = overall["valid_answers"]
    lines = [
        "# GRPO 预留 2000 题 LoRA 检索评测结果", "",
        f"整体严格 EM：**{overall['correct']} / 2000 = {overall['accuracy_final']:.2f}%**。",
        f"全部 2000 题已产生终态，pending 为 0；失败终态共 {overall['failed']} 题，计入分母并记 0 分。", "",
        "| 范围 | 正确数 / 题数 | 严格 EM | 答案词 F1（0–1） | 有效终答数 / 题数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, values in (("整体", overall), ("NQ", result["by_source"]["nq"]),
                          ("HotpotQA", result["by_source"]["hotpotqa"])):
        lines.append(f"| {label} | {values['correct']} / {values['selected']} | {values['accuracy_final']:.2f}% | "
                     f"{values['f1']:.4f} | {values['valid_answers']} / {values['selected']} |")
    lines.extend([
        "", f"整体有效终答比例为 **{valid} / 2000 = {100 * valid / 2000:.2f}%**。有效终答表示流程成功结束并解析出答案，不等于答案正确。",
        "EM 使用来源对应的官方规范化，与原黄金答案别名做完整相等比较；F1 是词重合指标。HotpotQA 使用官方答案 F1，NQ 的词 F1 为辅助指标。sub-EM 不计入准确率。", "",
        "## 失败分类", "", "| 类别 | 题数 |", "| --- | ---: |",
    ])
    failures = overall.get("error_categories", {})
    if failures:
        for key, value in sorted(failures.items()):
            lines.append(f"| {ERROR_NAMES.get(key, key)} | {value} |")
    else:
        lines.append("| 未记录失败终态 | 0 |")
    lines.extend([
        "", f"平均搜索次数：{overall['mean_search_count']:.3f}；平均生成 token：{overall['mean_generated_tokens']:.2f}。两项均以全部 2000 个终态为分母，包含失败题。", "",
        "## 模型与题集", "",
        f"- 基座：`{config['model_path']}`。",
        f"- LoRA adapter：`{config['adapter_path']}`。",
        f"- 原始题集：`{config['data_file']}`。",
        "- 题集为用户指定的 GRPO 预留集，NQ 1000 + HotpotQA 1000，原始来源划分为 train；未参与本次 SFT，原 partition=grpo 保持不变。",
        "- 本结果属于该预留集，不作为 NQ 官方 test 或 HotpotQA 官方 dev 成绩。黄金答案不进入模型，也不按预测临时补充答案别名。",
    ])
    if Path(config["adapter_path"]).name == "checkpoint320_best_lora":
        lines.append("- 本次使用完整 checkpoint-320 的 adapter 快照：320 次训练更新及 checkpoint 已保存，随后 NCCL 收尾等待 600 秒超时退出；缺少 final_adapter 目录不影响这些已存权重的加载。")
    lines.extend([
        "- 本报告未包含同题训练前基线，不给出训练增益结论。", "",
        "## 推理协议", "",
        "学生使用原始 Base Search-R1 PROMPT，greedy 生成搜索请求；请求本地真实 E5 检索，将返回内容接入 information，再继续生成答案。没有套聊天模板，检索 query 由学生生成。", "",
        "| 参数 | 本次配置 |", "| --- | --- |",
        f"| 最多搜索次数 | {config.get('max_searches', 3)} |",
        f"| 每次检索 top-k | {config.get('topk', 3)} |",
        f"| 每段最多字符数 | {config.get('passage_chars', 1800)} |",
        f"| 上下文上限 | {config.get('max_context_tokens', 8192)} tokens |",
        f"| 单轮生成上限 | {config.get('max_new_tokens', 512)} tokens |",
        f"| 每题累计生成上限 | {config.get('max_total_generated_tokens', 2048)} tokens |",
        f"| 单 worker 初始 batch | {config.get('batch_size', 4)} |",
        "| 解码 | greedy，do_sample=False |",
        f"| 检索服务 | `{config.get('retriever_url', 'http://127.0.0.1:8000/retrieve')}` |", "",
        "四个独立单卡 worker 推理，无 DDP collective；OOM 时该次生成 batch 递归减半重试，单题仍 OOM 则记录错误并保留其他待完成题。本结果取自当前配置绑定的正式输出目录，未混入旧 batch 4 v1 的 partial 预测。", "",
        "## 文件与原始轨迹", "",
        f"- 完整报告：`{output / 'report.json'}`。",
        f"- 精简逐题结果：`{output / 'predictions.compact.jsonl'}`，按原始 GRPO 2000 题顺序排列。",
        f"- 配置及数据/权重/代码校验记录：`{output / 'run_manifest.json'}`。",
    ])
    for worker in range(4):
        lines.append(f"- worker {worker} 完整原始轨迹：`{output / f'worker-{worker}.predictions.jsonl'}`。")
    lines.extend(["", f"本次汇总时间：`{result['updated_at']}`。", ""])
    text = "\n".join(lines)
    notice = output / "score_interpretation_notice.md"
    if notice.is_file():
        text = notice.read_text(encoding="utf-8").rstrip() + "\n\n" + text
    return text


def export(config):
    final_report = report(config)
    require_complete(final_report)
    examples = load_grpo(config["data_file"])
    records = compact_predictions(config, final_report, examples)
    output = Path(config["output_dir"])
    compact = output / "predictions.compact.jsonl"
    summary = output / "summary.md"
    text = summary_markdown(config, final_report)
    atomic_text(compact, "".join(canonical(row) + "\n" for row in records))
    atomic_text(summary, text)
    print(canonical({"已导出题数": len(records), "正确题数": final_report["overall"]["correct"],
                     "严格EM百分比": final_report["accuracy_final"],
                     "精简预测": str(compact), "中文总结": str(summary),
                     "完整报告": str(output / "report.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    export(load_config(args.config))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(f"结果导出失败：{error}", file=sys.stderr)
        sys.exit(1)
