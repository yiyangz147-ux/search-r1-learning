# 环境与配置

## 配置是示例

这个代码副本将具体部署路径替换为 `/srv/search-r1` 等示例路径，将GPU UUID替换为示例值，
将W&B entity替换为 `your-wandb-entity`。数据 SHA 的全零值也是占位符。
运行前需要填写自己的资源信息，并重新执行数据／模型准备和指纹检查。
保留严格校验，不应通过关闭检查来接受错误模型或不同数据。

检查这些位置：

- `configs/*.json`：SFT、评测输入、输出目录与模型路径。
- `configs/search_grpo_v1.yaml`：GRPO 数据、模型、预算、W&B 与训练配置。
- `scripts/run_search_grpo.py`：服务器根目录、GPU UUID 映射、已确认的实验约束。
- Shell 启动入口：Python 解释器、项目根目录和可读超参数。

## 依赖分环境安装

数据转换使用 `requirements-data.txt`；教师生成使用 `requirements-sft-runtime.txt`；
LoRA SFT 使用 `requirements-lora-sft.txt`。CUDA 依赖需要与自己的 Linux／GPU 环境匹配。
不应在正在运行的推理或训练环境中直接升级依赖。

数据工具可单独安装：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python scripts/prepare_data.py --help
```

公共数据由使用者通过下载脚本另行获取。下载后生成的 `data/` 继续由 Git 忽略。
模型、索引和训练需要的磁盘空间应另行核算，与代码仓库大小无关。

## GRPO 的上游源码

GRPO 启动器针对独立的 verl v0.6.1 checkout，补丁校验固定的上游文件 SHA。
完整上游框架不打包进本仓库，可从官方来源准备：

```bash
git clone --branch v0.6.1 https://github.com/volcengine/verl.git vendor/verl-0.6.1
python3 scripts/patch_search_grpo_runtime.py --vendor-root vendor/verl-0.6.1
```

新克隆环境若要运行依赖上游真实函数的测试，需要准备该 checkout 和相应Python依赖，
并把 `SEARCH_GRPO_TEST_VERL_ROOT` 指向它。部分真实模型／CUDA回归还需要PyTorch、Transformers和GPU。

## 外部服务与凭证

准备自己的推理服务、Wikipedia检索服务以及必要的教师API／W&B认证。
凭证通过交互式登录、环境变量或仓库外的私密文件提供，避免写进配置。
配置示例中的 localhost 地址表达的是服务接口约定，需要与自己的实际服务匹配。

## 上传前检查

```bash
git status --short
git diff --cached --stat
python3 scripts/verify_code_only.py
```

默认检查暂存区的真实内容；`--filesystem` 可在初始化Git之前检查整个干净副本。
全部检查通过后仍应人工阅读暂存差异，再提交和推送。
