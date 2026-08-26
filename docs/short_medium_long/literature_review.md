# 文献调研：within-question length-ranked reasoning distillation

更新日期：2026-08-26

## 1. 调研结论

有人做过非常接近的核心操作，但尚未在本次检索中发现与本项目完整设计相同的工作。

最直接的近邻是 [ShorterBetter](https://arxiv.org/abs/2504.21370)。它把同题多次 rollout 中的“最短正确响应长度”定义为 Sample Optimal Length，并用该长度构造 GRPO 奖励。[Implicit Compression Regularization](https://arxiv.org/abs/2605.07316) 也在 on-policy rollout group 内强化 shortest-correct samples；[BFS-PO](https://arxiv.org/abs/2602.14917) 则通过搜索和 RL 寻找逐渐变短的正确解答。

因此，以下说法不成立：

- “首次从多次采样中选择最短正确轨迹”；
- “首次发现短正确轨迹可以作为训练信号”；
- “shortest correct 就是每道题的真实最优推理长度”。

本项目仍然有清晰、可检验的差异化定位：现有近邻主要优化一个 policy 的推理效率，或者通过重写、截断和长度惩罚压缩轨迹；本项目使用离线 teacher-to-student sequence-level SFT，把同一个 answer-blind teacher 候选池中的 short、lower-median、long 三类完整且答案正确的轨迹作为三个受控训练条件，在相同问题支持集与相同 student 配方下比较下游行为。这个设计可以回答“student 应该模仿 teacher 分布中的哪个相对长度区域”，而不仅是“如何让一个 reasoning model 变短”。

这里仍不能把结果解释为 token 长度的纯因果效应。short 和 long 轨迹同时可能在策略、冗余、自检、表达风格和错误风险上不同。paired rewrite 与随机正确候选控制是区分这些机制所必需的。

## 2. 本项目方法的精确定义

对问题 `x` 和固定 teacher `T`：

1. 在不显示 gold answer、也不加入长度要求的同一 prompt 下采样 `K` 个候选；
2. 用可验证的最终答案过滤错误候选；
3. 对正确候选去重，并按 student tokenizer 下的 completion token 数排序；
4. 在同一题内部选择 short、lower-median 和 long 三个完整轨迹；
5. 只保留三个 rank 均可定义的 common problem support；
6. 固定 student、LoRA 配方和训练 seed，分别进行 completion-only SFT；
7. 分别报告准确率、生成长度、max-token hit、答案抽取失败和训练监督 token 总量。

当前实现使用 `K=16`、Qwen2.5-7B-Instruct teacher、Qwen2.5-1.5B-Instruct student、881 道 common-support GSM8K 题和 seed 17。方法属于黑盒、离线、sequence-level response distillation，不是 logit-level KD。

实现与证据：

- [生成配置](../../configs/capacity_length_ranked_sampling_7b_v1.json)
- [rank selection 实现](../../src/length_budget_distill/ranked_sampling.py)
- [训练配置](../../configs/capacity_length_ranked_sampling_7b_training_seed17_v1.json)
- [评测配置](../../configs/capacity_length_ranked_sampling_7b_eval_seed17_v1.json)
- [正式实验报告](../../results/capacity_length_ranked_sampling_7b_v1/formal/analysis/experiment_report.md)

## 3. 最相关工作

### 3.1 直接使用 shortest-correct 信号

| 工作 | 核心做法 | 与本项目相同处 | 关键差异 |
|---|---|---|---|
| [ShorterBetter, 2025](https://arxiv.org/abs/2504.21370) | 多次 rollout，定义最短正确响应的长度为 Sample Optimal Length，用正确性与距该长度的偏差构造 GRPO reward | 同题多采样、正确性过滤、shortest-correct 统计量 | on-policy RL；优化同一个 reasoning policy；只锚定 shortest-correct 长度，不做 teacher-to-student short/median/long SFT factorial |
| [Implicit Compression Regularization, 2026](https://arxiv.org/abs/2605.07316) | 从当前 policy 的 rollout group 中选择 shortest-correct samples，作为 RL 的隐式压缩正则 | 同题 rollout group 和 shortest-correct 选择 | on-policy RL regularizer；没有离线 teacher/student 分离，也没有 medium/long 对照 |
| [BFS-PO, 2026](https://arxiv.org/abs/2602.14917) | 用 best-first/backtracking search 寻找更短正确答案，并通过 RL 学习简洁推理 | 正确且短是训练目标 | 搜索式、逐步压缩和 RL；不是从固定随机候选池做 rank-matched SFT |
| [BIRD, 2026](https://arxiv.org/abs/2607.15736) | 在 brevity instruction 下采样，保留 answer-correct traces，先做 prompt-switch SFT，再做 on-policy reverse-KL | 正确性过滤、SFT、训练后默认更简洁 | 简洁性来自另一条 brevity prompt；没有在原 prompt 的同一候选分布中比较 short/median/long |

ShorterBetter 是最需要正面讨论的工作。它的完整方法说明其 rollout group size 为 8，SOL 作为动态 reward target 随 policy 更新。本项目当前候选池来自固定 7B teacher，数据在训练前一次性冻结；这使本项目更适合研究监督数据构造与 teacher/student capacity mismatch，但不具备 ShorterBetter 的 on-policy 自适应能力。

### 3.2 通过重写、难度条件或截断获得短轨迹

| 工作 | 数据或训练策略 | 与本项目的关系 |
|---|---|---|
| [LS-Mixture SFT, 2025](https://arxiv.org/abs/2505.03469) | 把 long CoT 做 structure-preserved rewriting，再混合 long/short 轨迹 SFT | 同属 teacher-to-student SFT，且显式研究 long/short；但 short 是 long 的重写版本，不是同一 prompt 下独立采样后的相对 rank |
| [LiteCoT, 2025](https://arxiv.org/abs/2505.19716) | teacher 先判断题目难度，再把 long trace 重写到适当长度，构建 100K concise traces | 研究 student capacity 和 difficulty-aware length；但长度来自 difficulty judgment 与 rewrite，不是 within-question order statistic |
| [Less is More Tokens, 2025](https://arxiv.org/abs/2509.05226) | 构造与问题难度成比例的 CoT，结合 SFT 与 DPO | 支持“推理长度应依题目变化”的研究问题；没有 short/median/long common-support factorial |
| [Distilling the Essence, 2025/2026](https://arxiv.org/abs/2512.21002) | 比较 prompt、CoT、answer 各段监督，并截断训练序列 | 研究训练 token 与性能的关系；截断可能移除后半段，和选择完整正确轨迹不同 |
| [BRIDGE, 2026](https://arxiv.org/abs/2602.17686) | structure-aware masking、GRPO 和失败样本重写，缓解 verbose teacher 与小 student 的容量差 | 与 capacity mismatch 问题直接相关；训练机制与本项目的固定 SFT 对照不同 |

这些工作表明“较短监督有时更有效”已经是活跃方向，但数据生成方式不同。对本项目最重要的机制对照是已有 [paired-rewrite protocol](../capacity_length_paired_rewrite_7b_pilot_protocol.md)：它可以比较“独立采到的短解”与“同一长解的压缩版本”，从而判断收益更像来自路径选择还是冗余删除。

### 3.3 多样采样、正确性过滤和一般 rationale distillation

| 工作 | 已建立的结论 | 与本项目的边界 |
|---|---|---|
| [Self-Consistency, 2022/ICLR 2023](https://arxiv.org/abs/2203.11171) | 推理时采样多条 reasoning path，并对答案做 marginalization/majority aggregation | 奠定同题多样采样，但不选择轨迹做 student SFT，也不研究长度 rank |
| [STaR, 2022](https://arxiv.org/abs/2203.14465) | 生成 rationale，保留最终答案正确的样本并迭代微调 | 奠定 correctness-filtered self-training，但没有长度选择 |
| [ReST-EM, 2023/2024](https://arxiv.org/abs/2312.06585) | 采样、二值反馈过滤、微调并重复 | 与候选生成和正确性过滤接近，但优化信号不包含 within-question length rank |
| [Distilling Step-by-Step, 2023](https://arxiv.org/abs/2305.02301) | 使用 LLM rationale 作为额外监督训练更小模型 | 奠定 rationale distillation，但没有多候选长度处理 |
| [DeepSeek-R1, 2025](https://arxiv.org/abs/2501.12948) | 用 RL 激励 reasoning，并把大模型的 reasoning pattern 蒸馏到小模型 | 证明 reasoning distillation 可扩展；没有本项目的相对长度受控比较 |

这组工作说明本项目的生成、过滤和 SFT 三个组件本身都已有充分先例。研究贡献必须来自它们的组合方式、控制设计和 capacity-by-length 结论，而不是把一般 rationale distillation 包装成新方法。

### 3.4 主要发生在 inference-time 的长度控制

[s1](https://arxiv.org/abs/2501.19393) 使用 budget forcing 在测试时截断或延长 thinking；[EDIT](https://arxiv.org/abs/2509.06174) 在测试时通过约束生成寻找简洁正确路径。它们适合成为 inference-time baselines，但不能回答不同长度的 teacher supervision 如何改变 student。

## 4. 数据质量相关证据

[Answer-Conditioned Chains of Thought Degrade Verifiable-Reasoning Distillation](https://arxiv.org/abs/2607.14552) 报告：把 gold answer 暴露给生成器后再要求其构造 reasoning，可能产生 correctness filter 无法发现的反向合理化。当前项目的 answer-blind generation 是正确选择；扩大实验时不应为了提高低容量 teacher 的正确候选覆盖率而把 gold answer 放入 prompt。

最终答案正确也不保证中间推理有效。大实验至少需要：

- 检测过早泄露最终答案、循环重复、自相矛盾和 512-token 截断；
- 对 teacher、rank、题目难度分层抽样做人工或独立 judge 审计；
- 同时报告最终答案 verifier 通过率和过程质量诊断，不把二者合并成一个指标；
- 保存完整候选池，使任何新的过滤器都能在不重新生成的情况下复算。

## 5. 建议的论文定位

### 5.1 可使用的表述

> We conduct a controlled study of within-question, length-ranked, correctness-filtered trajectory distillation. For each problem, we sample an answer-blind teacher repeatedly, freeze the correct trajectory pool, and train matched students on short-, median-, or long-ranked complete solutions under identical problem support and SFT settings.

> Our goal is not to propose the shortest-correct statistic, but to characterize when different regions of a teacher's correct length distribution are useful to students of different capacities.

中文可表述为：

> 本文不是提出 shortest-correct selection，而是研究 teacher 正确轨迹分布中的相对长度位置，在何种 teacher/student 容量组合与训练 token 约束下，最适合作为小模型的离线序列级监督。

### 5.2 应避免的表述

- “最短正确轨迹就是最优 reasoning”；
- “越短越好”；
- “首次 shortest-correct distillation”；
- “长度本身导致准确率提高”；
- “已证明可以泛化到数学推理以外的任务”；
- “single-seed GSM8K 结果已确认”。

更合理的中心假设是一个 capacity- and difficulty-dependent optimum：对容量较小的 student，过长 teacher 轨迹可能带来冗余与容量失配；对困难题或更强 student，过短轨迹也可能删除必要计算。因此预注册时应允许 short、medium 或 adaptive mixture 中任一条件胜出。

## 6. 检索范围与限制

本次检索于 2026-08-26 进行，优先核验 arXiv 原始论文页面和可用的论文 HTML。关键词覆盖 `shortest correct reasoning`, `multiple sampled correct traces`, `length-ranked reasoning distillation`, `long-short CoT SFT`, `difficulty-aware CoT distillation` 和 `reasoning sequence truncation`。

这不是法律意义或系统综述意义上的穷尽性 novelty search。尤其是 2025–2026 年 concise reasoning/RLVR 文献更新很快；投稿前应按标题、引用网络和最新会议 proceedings 再更新一次。当前结论应写成“在本次检索覆盖的工作中未发现完整相同设计”，而不是“没有任何人做过”。
