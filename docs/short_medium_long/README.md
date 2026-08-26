# Short–medium–long 轨迹蒸馏研究

更新日期：2026-08-26

## 结论摘要

这个方向值得扩大，但下一步不应直接把全部模型和全部条件做成一个大网格。当前结果最有价值的地方，是在同一道题、同一个无长度约束 prompt、同一个 teacher 候选池中，先过滤答案正确的轨迹，再选择相对 short、medium、long 的完整解答训练同一个 student。它比直接给 teacher 加 128/256/512 token 提示更好地控制了题目与 prompt 差异。

当前 revised formal single-seed 结果中，short-ranked SFT 的点估计最高，同时 student 的生成长度随训练轨迹由 short 到 long 单调增加。不过，short 与 long 的准确率差异经预注册多重比较校正后没有达到显著性，因此当前证据支持“轨迹长度可被 SFT 传递，并出现值得复现的准确率排序”，不支持“最短轨迹普遍最优”。

文献中已经有人使用“多次采样后取最短正确响应”的核心统计量。最直接的近邻是 [ShorterBetter](https://arxiv.org/abs/2504.21370)，其 Sample Optimal Length 就是多次 rollout 中最短正确响应的长度，并把它用于 GRPO 奖励。因此，本项目不能声称首创 shortest-correct selection。更稳健的论文定位是：

> 系统研究 answer-blind teacher 候选分布中的 within-question relative length rank，如何在固定问题支持集、固定 student 和固定 SFT 配方下影响小模型的准确率、推理长度与截断失败，并进一步测量 teacher capacity、student capacity、sampling density 和监督 token 总量的交互。

完整文档：

- [文献调研与论文定位](literature_review.md)
- [大规模实验设计](large_scale_experiment_plan.md)

## 当前证据锚点

| 条件 | 准确率 | 平均输出 token | 训练轨迹平均 token |
|---|---:|---:|---:|
| Base 1.5B | 67.14% | 242.6 | 不适用 |
| Short-ranked SFT | 70.29% | 264.1 | 194.4 |
| Medium-ranked SFT | 69.35% | 296.5 | 242.5 |
| Long-ranked SFT | 68.01% | 330.3 | 318.0 |

核心比较为 short 对 long：`+2.29 pp`，paired bootstrap 95% CI `[0.00, 4.57] pp`，Holm-adjusted McNemar `p=0.1747`。这是 revised formal single-seed evidence，不估计训练 seed 变异。

证据路径：

- [正式报告](../../results/capacity_length_ranked_sampling_7b_v1/formal/analysis/experiment_report.md)
- [完成审计](../../results/capacity_length_ranked_sampling_7b_v1/formal/completion_audit.json)
- [生成审计](../../results/capacity_length_ranked_sampling_7b_v1/formal/datasets/generation_audit.json)
- [配对比较](../../results/capacity_length_ranked_sampling_7b_v1/formal/analysis/paired_contrasts.csv)
- [主图 PNG](../../figures/capacity_length_ranked_sampling_7b_v1/formal/ranked_length_accuracy_and_output_length.png)
- [主图 PDF](../../figures/capacity_length_ranked_sampling_7b_v1/formal/ranked_length_accuracy_and_output_length.pdf)

## 推荐执行顺序

1. 先补 seed 42 和 73，复现当前 `7B teacher -> 1.5B student, K=16` 的三种 rank；这只需新增 6 个 adapter。
2. 从一个 `K_max=32` 的不可变候选池构造嵌套的 `K={4,8,16,32}` 数据，先做数据分布与支持集审计，再锁定大实验的 sampling density。
3. 在固定 1.5B student 上做 `teacher={1.5B,3B,7B,14B} x rank={short,medium,long} x seed={17,42,73}`，共 36 个 adapter；1.5B teacher 明确标为 self-distillation control。
4. 只有 teacher-by-rank 交互稳定后，再增加 student capacity，而不是一开始做所有交叉项。
5. 在少数关键端点补 equal-supervision-token、随机正确候选、prompt-constrained short 和 paired rewrite 控制，区分长度、内容策略和训练曝光量。

项目内 GSM8K official test 已全部被观察，[paired-rewrite confirmatory manifest](../../results/capacity_length_paired_rewrite_7b_pilot_v1/pilot/eval/confirmatory/eval_manifest_confirmatory_shard_00_of_01.json) 也表明 `train[3000:7473]` 已被用于评测。因此后续 GSM8K 结果应标为锁定比较性复现，不应称为新的 untouched confirmation。任何新的 OOD 确认性声明都需要另立并审批协议；当前文档没有扩大现有 GSM8K 论文声明范围。
