# Search-R1 Learning

用于学习 Qwen3-4B-Base 的 RAG、教师轨迹生成、LoRA SFT 和 GRPO 的代码仓库。
保留源代码、测试、配置示例和使用教程。数据集、Wikipedia、检索索引、模型权重、
adapter、训练轨迹、评测输出、日志、缓存和凭证均不在仓库中。

这是学习项目，参考 [Search-R1](https://github.com/PeterGriffinJin/Search-R1)。
Qwen3、LoRA、小样本数据规模和工具预算属于本项目的适配，不能把它当成官方论文数值复现。
配置中的路径、GPU UUID、W&B entity 和预期数据指纹均为示例，运行前需要配置自己的资源。

## 从哪里开始

1. 阅读 [Git 基础教程](docs/GIT_BASICS.md)，理解本地保存与上传 GitHub 的区别。
2. 阅读 [RAG → SFT → GRPO 流程](docs/RAG_SFT_GRPO.md)，理解各脚本的角色。
3. 阅读 [环境与配置](docs/CONFIGURATION.md)，准备依赖和自己的数据目录。

## 代码入口

| 工作 | 入口 |
|---|---|
| 下载并转换公共问答集、准备检索资源 | `scripts/prepare_data.py` |
| Base 模型有／无 RAG 对照评测 | `scripts/evaluate_rag.py` |
| 调用教师、执行搜索、筛选 SFT 轨迹 | `scripts/prepare_sft.py` |
| 清洗、划分和转换 SFT 输入 | `scripts/prepare_sft_splits.py`、`scripts/prepare_lora_sft_data_v2.py` |
| 完整 embedding 与 q/v LoRA SFT | `scripts/train_search_lora_sft_v2.py` |
| SFT 模型工具交互评测 | `scripts/evaluate_full_embedding_rag.py` |
| GRPO 数据和模型准备 | `scripts/prepare_search_grpo.py` |
| GRPO 检索／生成工具循环 | `scripts/search_grpo_agent_loop.py` |
| 答案 EM、格式分与审计 | `scripts/search_grpo_reward.py`、`scripts/search_grpo_metrics.py` |
| GRPO 分阶段启动 | `scripts/run_search_grpo.sh`、`scripts/run_search_grpo.py` |
| checkpoint 导出及最终配对评测 | `scripts/finish_search_grpo.py` |

`configs/` 包含配置示例，`tests/` 包含测试源代码。测试里的人工构造输入服务于程序断言；
仓库不包含真实训练题集或评测题集。`tests/fixtures/official_qa_metrics/` 保存的是官方
评测程序及许可证、来源说明，属于代码。

## 防止数据进入 Git

`.gitignore` 默认忽略所有路径，只开放明确的源码、配置、依赖和文档路径。
每次提交前，还应检查实际暂存内容：

```bash
git status --short
git diff --cached --stat
python3 scripts/verify_code_only.py
```

检查脚本读取 Git index 中的真实 blob，拒绝数据目录、非允许文件、二进制、大文件和常见凭证。
它是辅助检查，仍需要阅读 `git diff --cached`。仓库访问设为私有也不替代内容检查。

上游代码和依赖的来源见 [NOTICE](NOTICE) 及测试 fixture 中保留的许可证。
