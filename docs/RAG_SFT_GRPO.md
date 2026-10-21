# RAG → SFT → GRPO：每一步在做什么

## 1. RAG：先找到资料，再生成答案

输入问题后，检索器用 E5 把查询转换为向量，用 FAISS 从 Wikipedia 段落向量中寻找相近内容。
程序把返回的段落作为观察加入模型上下文，模型继续生成搜索动作或最终答案。
`prepare_data.py` 准备外部资源，`evaluate_rag.py` 连接已有模型和检索服务运行评测。

检索器与生成模型各有职责：E5 产生检索向量，Qwen 读取问题和证据、产生文本。
检索失败与生成失败应分开检查；只看最终 EM 很难定位问题。

## 2. 教师生成 SFT 轨迹

`prepare_sft.py` 调用教师产生工具动作，程序执行真实搜索并记录观察，再让教师继续。
一条轨迹包含问题、assistant 的动作、检索返回的 environment 观察及最终答案。
评分、格式校验和长度检查决定哪些轨迹可以进入训练。

本项目约定的动作标签是 `<think>`、`<search>`、`<information>`、`<answer>`。
应保留角色来源，避免把检索内容误当成模型生成的答案。

## 3. LoRA SFT：学习轨迹写法

SFT 让学生学习教师轨迹中的 assistant token。检索观察提供上下文，训练 mask 决定哪些位置进入 loss。
`search_sft_dataset.py` 组织输入与 mask，训练入口读取配置并更新指定参数。
完整 embedding 与共享输出权重的处理在 `full_embedding_lora.py` 中实现；仅加载 adapter 可能不足以恢复完整 SFT 模型。

## 4. GRPO：从同题采样中学习

GRPO 对同一道题采样多条轨迹，以答案 reward 比较同组轨迹，计算相对 advantage，
再用带 clip、KL 和熵项的目标更新策略。工具观察仍作为上下文保留，但不计入 assistant 的 policy loss。

`search_grpo_agent_loop.py` 管理生成、搜索与预算；`search_grpo_reward.py` 给答案评分；
`search_grpo_metrics.py` 记录独立的答案 reward、格式分和 mask 审计；
`run_search_grpo.py` 执行检查、smoke、正式训练及最终导出评测。

配置示例以答案 EM 为优化 reward，格式分单独记录。训练采样 reward、验证 greedy EM
和最终测试 EM 含义不同，应分别阅读。

## 5. 分块概率计算

Qwen 的大词表使长轨迹上的 log probability 和 entropy 计算占用大量临时显存。
当前 padded SDPA 适配按128个token分块计算这些统计量，并逐块 checkpoint。
模型前向仍读取完整上下文；分块发生在输出 logits 之后，保持原生概率／熵公式和 loss mask。

## 6. 保持评测可比较

训练集用于参数更新，独立验证集用于选 checkpoint，最终测试集在选模结束后使用。
比较模型时保持问题集合、prompt、工具预算、检索配置和答案评分规则一致。
权重合并或推理实现变化可能影响数值与生成，应保留可复核的对照，而不是只比较两个汇总百分比。

数据下载、模型权重和实际运行结果由使用者自行准备，均不包含在这个代码仓库中。
