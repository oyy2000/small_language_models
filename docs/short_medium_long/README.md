# Short–medium–long 轨迹蒸馏研究

更新日期：2026-08-28

## 结论摘要

这个方向值得扩大，但下一步不应直接把全部模型和全部条件做成一个大网格。当前结果最有价值的地方，是在同一道题、同一个无长度约束 prompt、同一个 teacher 候选池中，先过滤答案正确的轨迹，再选择相对 short、medium、long 的完整解答训练同一个 student。它比直接给 teacher 加 128/256/512 token 提示更好地控制了题目与 prompt 差异。

当前三训练 seed 比较性复现中，short-ranked SFT 的平均准确率最高，同时 student 的生成长度随训练轨迹由 short 到 long 单调增加。short 对 long 的 crossed seed/problem bootstrap 区间为正，且三项预注册比较做 Holm 校正后达到 `p=0.0462`。这支持“在当前 7B teacher、1.5B student 和已观察 GSM8K cohort 上，short-ranked supervision 的优势可跨三个训练 seed 复现”，但不证明轨迹长度本身是纯因果变量，也不支持跨 teacher、student 或 benchmark 的普遍最优声明。

文献中已经有人使用“多次采样后取最短正确响应”的核心统计量。最直接的近邻是 [ShorterBetter](https://arxiv.org/abs/2504.21370)，其 Sample Optimal Length 就是多次 rollout 中最短正确响应的长度，并把它用于 GRPO 奖励。因此，本项目不能声称首创 shortest-correct selection。更稳健的论文定位是：

> 系统研究 answer-blind teacher 候选分布中的 within-question relative length rank，如何在固定问题支持集、固定 student 和固定 SFT 配方下影响小模型的准确率、推理长度与截断失败，并进一步测量 teacher capacity、student capacity、sampling density 和监督 token 总量的交互。

完整文档：

- [文献调研与论文定位](literature_review.md)
- [大规模实验设计](large_scale_experiment_plan.md)

## 当前证据锚点

| 条件 | 三-seed平均准确率 | 平均输出 token | 训练轨迹平均 token |
|---|---:|---:|---:|
| Base 1.5B | 67.14% | 242.6 | 不适用 |
| Short-ranked SFT | 70.32% | 263.9 | 194.4 |
| Medium-ranked SFT | 69.58% | 295.1 | 242.5 |
| Long-ranked SFT | 67.59% | 331.1 | 318.0 |

核心比较为 short 对 long：三 seed 平均 `+2.73 pp`，crossed seed/problem bootstrap 95% CI `[+0.60,+4.89] pp`，Holm-adjusted bootstrap `p=0.0462`。三个 seed 的差异分别为 `+2.29/+2.60/+3.31 pp`。这是对已观察 GSM8K `test[50:1319]` 的比较性复现，独立评测问题仍是 1,269 道，不是 3,807 道。

证据路径：

- [三-seed正式报告](../../results/capacity_length_ranked_sampling_7b_multiseed_v1/formal/analysis/experiment_report.md)
- [三-seed完成审计](../../results/capacity_length_ranked_sampling_7b_multiseed_v1/formal/completion_audit.json)
- [三-seed完成标记](../../results/capacity_length_ranked_sampling_7b_multiseed_v1/MULTISEED_COMPLETE)
- [生成审计](../../results/capacity_length_ranked_sampling_7b_v1/formal/datasets/generation_audit.json)
- [聚合配对比较](../../results/capacity_length_ranked_sampling_7b_multiseed_v1/formal/analysis/aggregate_paired_contrasts.csv)
- [主图 PNG](../../figures/capacity_length_ranked_sampling_7b_multiseed_v1/formal/ranked_multiseed_accuracy_and_output_length.png)
- [主图 PDF](../../figures/capacity_length_ranked_sampling_7b_multiseed_v1/formal/ranked_multiseed_accuracy_and_output_length.pdf)

## 推荐执行顺序

1. Phase A 已完成：新增 seed 42 和 73 的 6 个 adapter，并与 seed 17 合并完成三-seed评测、统计和审计。
2. 用户批准的 Phase C 主矩阵已冻结并提交：固定 `K=16`，在 1.5B student 上做 `teacher={1.5B,3B,7B,14B} x rank={short,medium,long} x seed={17,42,73}`，共 36 个 adapter；1.5B teacher 明确标为 self-distillation control。[冻结协议](../../results/capacity_length_ranked_sampling_multiteacher_v1/formal/protocol/frozen_protocol.json)和[提交清单](../../results/capacity_length_ranked_sampling_multiteacher_v1/formal/protocol/submission_manifest.json)已绑定配置、九作业 DAG、替换记录和清单记录时的工作树源码哈希；[清单注记](../../results/capacity_length_ranked_sampling_multiteacher_v1/formal/protocol/submission_manifest_annotation.json)进一步限定哈希证据范围并记录 BeeGFS 迁移。训练前审计还发现原 `index % 3` 顺序会把 seed 与 C30/C31/C32 完全混杂，因此新增了不改变 36 个实验 cell 的[平衡 launcher plan](../../results/capacity_length_ranked_sampling_multiteacher_v1/formal/protocol/launcher_assignment_plan.json)：每个节点各 12 个 run、每个 teacher 3 个、每个 rank 4 个、每个 seed 4 个，且同一 teacher-rank cell 的三个 seed 分布到三个节点；每个节点的三-run wave 同时含三个不同 teacher 和 short/medium/long 各一个，跨节点的同一 wave 也平衡三个 seed。提交清单只证明调度来源，不是完成证据。当前尚无主矩阵性能结果。
3. `K_max=32` 与嵌套 `K={4,8,16,32}` 的 Phase B 数据审计保留为后续 sampling-density robustness，不得与当前 `K=16` 主矩阵混报。
4. 只有 teacher-by-rank 交互稳定后，再增加 student capacity，而不是一开始做所有交叉项。
5. 在少数关键端点补 equal-supervision-token、随机正确候选、prompt-constrained short 和 paired rewrite 控制，区分长度、内容策略和训练曝光量。

项目内 GSM8K official test 已全部被观察，[paired-rewrite confirmatory manifest](../../results/capacity_length_paired_rewrite_7b_pilot_v1/pilot/eval/confirmatory/eval_manifest_confirmatory_shard_00_of_01.json) 也表明 `train[3000:7473]` 已被用于评测。因此后续 GSM8K 结果应标为锁定比较性复现，不应称为新的 untouched confirmation。任何新的 OOD 确认性声明都需要另立并审批协议；当前文档没有扩大现有 GSM8K 论文声明范围。
