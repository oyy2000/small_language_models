# Short–medium–long 大规模实验设计

更新日期：2026-08-28

## 1. 研究目标

把当前单 teacher、单 student、单 seed 的结果扩大成一个可解释的研究，而不是单纯增加 adapter 数量。核心问题为：

> 在最终答案正确的前提下，teacher 同题候选分布中的相对轨迹长度，如何与 teacher capacity、student capacity、sampling density、问题难度和监督 token 总量共同决定 student 的准确率与推理成本？

主研究仍限定为 GSM8K。MATH-500 和其他 OOD benchmark 只能在单独协议获批后作为 evaluation-only 扩展；不得把 MATH-500 加入训练，也不得在当前方案中静默扩大论文声明范围。

## 2. 当前起点与证据边界

当前实验是：

- teacher：Qwen2.5-7B-Instruct；
- student：Qwen2.5-1.5B-Instruct，rank-4 LoRA；
- sampling：每题 16 个候选，temperature 0.7，top-p 0.95，最大 512 new tokens；
- selection：正确候选精确去重后取 shortest、lower-median、longest；
- support：881 道固定 common-support GSM8K 训练题；
- training：equal-example，completion-only SFT，training seed 17、42、73；
- evaluation：已锁定且已观察的 GSM8K `test[50:1319]`，1269 题，greedy，最大 512 new tokens。

生成审计记录了 14,096 个预期和实际候选，其中 13,962 个通过最终答案 verifier；881 道题全部满足至少 3 个唯一正确候选，三个 rank 使用相同问题支持集。训练监督 token 总量分别为 171,252、213,621 和 280,168，因此 equal-example 不等于 equal-supervision-token。

Phase A 的正式结果与限制见 [三-seed报告](../../results/capacity_length_ranked_sampling_7b_multiseed_v1/formal/analysis/experiment_report.md)。完成审计为 passed：9 个 adapter、10 个评测模型和 12,690 条 prediction 均完整。short 对 long 的三-seed平均差为 `+2.73 pp`，crossed seed/problem bootstrap 95% CI `[+0.60,+4.89] pp`，三项比较 Holm 校正后 `p=0.0462`。该 cohort 已被观察，因此证据级别仍是比较性复现，不是新的 untouched confirmation。

## 3. 预注册假设

建议把“short 一定最好”改为以下可证伪假设：

- H1，长度传递：训练轨迹的相对 rank 会在固定解码条件下单调改变 student 平均输出长度和 max-token hit rate。
- H2，容量交互：准确率最优 rank 取决于 teacher/student capacity ratio；小 student 更可能受益于较短的正确监督，但不是预设必然结果。
- H3，难度交互：简单题的较短轨迹更可能处于 accuracy–length Pareto frontier，困难题可能需要 medium 或 long 轨迹。
- H4，sampling density：`K` 增加会机械性地缩短 minimum、拉长 maximum；如果不控制 `K`，所谓 length effect 会混入 order-statistic extremity。
- H5，训练曝光：equal-example 与 equal-supervision-token 可能产生不同排序，因为二者分别控制样本数和原始 completion token，而不同时控制 optimizer steps。
- H6，路径与压缩机制：独立采到的 short correct trajectory 和从 long trajectory 重写出的短版本可能具有不同训练效果，说明长度 rank 同时编码了解题策略选择。

准确率是 primary endpoint。输出长度、max-token hit、抽取失败、延迟和 token 使用量分别作为 secondary diagnostics，不建议在主分析中合成为一个可调权重的单一分数。

## 4. 数据构造协议

### 4.1 不可变候选池

大实验应一次生成 `K_max=32` 的 answer-blind 候选池，然后用 candidate index 前缀构造嵌套的 `K={4,8,16,32}`。这样 sampling-density 分析复用同一批随机候选，避免不同 `K` 同时改变随机 realization。

每条候选必须保存：

- problem ID、teacher ID、generation seed、candidate index 和原始 prompt；
- 原始文本、teacher tokenizer token 数和 student tokenizer token 数；
- 抽取答案、gold answer、verifier 结果与 verifier 版本；
- stop reason、是否命中最大长度、文本 hash 和配置 hash；
- shard ID、模型 revision 和生成代码 hash。

原始候选只能追加，不能在后续过滤时覆盖。ranked dataset 应由冻结候选池确定性派生。

生成规模公式为：

```text
candidate_count = number_of_teachers * number_of_source_problems * K_max
completion_tokens ~= candidate_count * mean_completion_tokens
```

例如 4 个 teacher、2,000 道 source problems、`K_max=32` 对应 256,000 个候选；若平均 completion 为 250 tokens，约为 64M generation tokens。这个数字是容量规划近似，不是运行时估计。

### 4.2 rank 定义

当前 `K=16` 的 `minimum/lower-median/maximum` 必须保留为 exact replication branch。大规模主分析更建议使用固定经验分位数，如 `q10/q50/q90`，理由是 minimum 和 maximum 随 `K` 系统变化，容易把 sampling density 与目标长度混在一起。

建议同时保留两种 estimand：

- extreme-rank estimand：复现当前 minimum/lower-median/maximum，直接回答扩大候选池后极端轨迹的效果；
- quantile-rank estimand：固定 `q10/q50/q90`，回答 teacher 正确分布的不同区域对 student 的效果。

两者不能在看过 formal accuracy 后再择优报告。应在 development 数据上锁定一个作为 primary，另一个作为 robustness analysis。

### 4.3 正确性、去重与支持集

最终答案 verifier 是入门门槛，不是完整过程 verifier。建议增加：

- 数学表达与答案格式标准化；
- 近重复检测，避免只有措辞变化的多个候选虚增有效 `K`；
- 过早出现 gold answer、循环段落、自相矛盾、异常 stop 和截断诊断；
- teacher x rank x difficulty 分层的过程质量审计。

不得在候选不足时向 teacher 显示 gold answer。相关工作已指出 answer-conditioned reasoning 可能产生 correctness filter 无法识别的反向合理化。

支持集需要两层定义：

- within-teacher rank effect：对每个 teacher 使用该 teacher 三个 rank 的 common support；这是主要、保留数据最多的估计。
- cross-teacher interaction：只在所有 teacher 和所有 rank 的 global common support 上比较 teacher capacity；同时报告 retention rate 和被筛除问题的难度偏差。

只报告 global intersection 可能把数据强烈过滤为容易题；只报告各 teacher 自己的 support 又会混入题目组成差异。两种分析都需要，且问题 ID 必须 hash-bound。

### 4.4 控制条件

除了 short、medium、long，优先增加以下控制，而不是盲目增加更多 rank：

1. random-correct：从同一正确候选池随机取一条，并冻结 selection seed；估计 rank selection 相对普通 rejection sampling 的价值。
2. hard-budget prompt：与历史 128/256/512 prompt-conditioned traces 对比，区分相对选择和显式长度提示。
3. paired rewrite：把同一 long trace 重写到目标比例，区分“换了一条更简洁的解题路径”和“删除原路径冗余”。
4. equal-supervision-token：对选定端点做 token-balanced robustness analysis。
5. adaptive mixture：只在前述效应稳定后，根据预先定义的题目难度路由到 short/medium/long；不得用 formal test 选择路由规则。

## 5. 分阶段扩大方案

### Phase A：多 seed 复现当前结果

状态：已于 2026-08-28 完成训练、锁定评测、统计、图表与 completion audit。

固定当前所有配置，只增加 training seeds 42 和 73。

| 项目 | 数量 |
|---|---:|
| 已有 adapter | 3 个 rank x seed 17 = 3 |
| 新增 adapter | 3 个 rank x 2 个 seed = 6 |
| 完成后总数 | 9 |

进入下一阶段的 gate 不要求 short 必须显著胜出，而要求：

- 所有 adapter、prediction 和 manifest 完整；
- 三个 seed 下输出长度方向可复现，或明确记录不稳定性；
- 估计 short–long accuracy difference 的 seed 间方差；
- 所有 pairwise prediction 保持同一评测问题顺序和支持集；
- 不根据已观察的 GSM8K official test 调学习率或 epoch。

### Phase B：sampling-density 与 rank-definition 数据审计

对 7B teacher 生成 `K_max=32`，在不训练 student 的情况下分析 `K={4,8,16,32}`：

- 正确候选数与唯一候选数；
- minimum、q10、median、q90、maximum 的长度分布；
- common-support retention；
- rank 间策略相似度和过程质量；
- candidate-pool seed 的 bootstrap 稳定性。

在 development 规则下锁定 `K` 与 primary rank definition。只有数据分离度、支持率和质量审计都通过才训练这一阶段的 adapter。这样可以避免先跑 36 个 adapter，再发现 minimum 只是偶然的极短异常值。

### Phase C：teacher capacity x rank 主因子

状态：用户批准固定当前 `K=16` 直接执行该阶段；协议已于 2026-08-28 冻结并提交 generation DAG。[提交清单](../../results/capacity_length_ranked_sampling_multiteacher_v1/formal/protocol/submission_manifest.json)记录了当前九作业 DAG、两个被替换 generation 作业、修复后的依赖关系和清单记录时的工作树源码哈希；[清单注记](../../results/capacity_length_ranked_sampling_multiteacher_v1/formal/protocol/submission_manifest_annotation.json)明确这些哈希不是 Slurm spool 快照，并记录正式结果根迁移到 BeeGFS 的校验过程。训练前发现朴素 product order 加 `index % 3` 会把 seed 与节点混杂；[平衡 launcher plan](../../results/capacity_length_ranked_sampling_multiteacher_v1/formal/protocol/launcher_assignment_plan.json)在不改变 36 个注册 cell 或训练超参数的前提下，让每个 C30/C31/C32 shard 各包含 12 个 run，并平衡 teacher、rank、seed 边际。每个节点的三-run wave 包含三个不同 teacher 和三个 rank，跨节点同一 wave 对三个 seed 和三个 rank 各有三个 run，从而同时降低节点与运行时段混杂。该清单的证据类别是 `submission_provenance_not_completion_evidence`。当前尚无 Phase C 性能结果，不能把排队任务当作主矩阵证据。Phase B 保留为后续 sampling-density robustness。

固定 1.5B student：

```text
teacher = {Qwen2.5-1.5B, 3B, 7B, 14B-Instruct}
rank = {short, medium, long}
training_seed = {17, 42, 73}
```

共 `4 x 3 x 3 = 36` 个 adapter。1.5B teacher 条件必须称为 self-distillation control。所有 teacher 保持相同的 answer-blind prompt、sampling hyperparameters、`K` 和最大生成长度。

这一阶段回答 paper 的主问题：更强 teacher 是否产生对小 student 更有用的短轨迹，以及长轨迹的容量失配是否随 teacher/student gap 增大。

### Phase D：student capacity moderator

Phase C 显示稳定 teacher-by-rank 交互后，再扩 student：

```text
teacher = {7B, 14B}
student = {0.5B, 1.5B, 3B}
rank = {short, medium, long}
training_seed = {17, 42, 73}
```

完整矩阵为 54 个 adapter，其中 18 个 1.5B-student cells 可从 Phase C 复用，因此新增 36 个。这个阶段必须使用独立 config/result/checkpoint roots，因为现有 factorial protocol 固定 1.5B student。

如果资源有限，优先选择 Phase C 中 rank 差异最大的一个 teacher 和最小差异的一个 teacher，而不是依据哪个条件准确率最高来挑选。

### Phase E：公平性与机制 ablation

只对预注册的关键端点运行：

- equal-example 对 equal-supervision-token；
- random-correct 对 rank-selected；
- independent short sample 对 paired rewrite short；
- greedy 512-token evaluation 对固定 budget sweep；
- 必要时增加 5 个 training seeds，而不是继续增加未经解释的模型条件。

equal-token 的含义必须保持为原始 completion supervision token 总量相等。它不会自动使样本数、optimizer steps、scheduler exposure 或信息密度相等，这些量都要单独报告。

### Phase F：单独审批的 OOD 扩展

当前协议不授权 OOD paper claim。若后续批准，应新建独立 protocol 并做到：

- MATH-500 evaluation-only，不能用于训练、selection 或超参数调整；
- 把 GSM8K、MATH-500 及其他 benchmark 分栏报告，不能混成一个主准确率；
- 在任何 OOD prediction 生成前冻结模型条件、解码参数、primary contrasts 和 multiplicity family；
- 明确哪些 benchmark 已被模型/研究者观察，哪些可作为新的 confirmation。

## 6. 评测与统计分析

### 6.1 当前数据的功效提示

原始 seed-17 short 对 long 在 1269 题中有 124 个 `short-only correct` 和 95 个 `long-only correct`，观察差异为 2.285 pp。用这组 discordance 做 McNemar 正态近似规划，在效应不缩水的乐观前提下：

- 双侧 `alpha=0.05`、80% power 约需 2,592 道独立配对题；
- 把三项 primary contrasts 的最保守阈值近似为 `alpha=0.05/3` 时约需 3,457 道独立配对题。

这是基于原始单-seed已观察效应的历史近似规划，不是保证。新的三-seed平均 short–long 差为 2.73 pp，但没有增加独立问题数，因此不能把 3,807 条 seed-question prediction 当作 3,807 道独立题重新计算 power。正式设计仍应做 1、2、3 pp 的 sensitivity curve。

GSM8K official test 的 1,319 题已全部被项目观察，[paired-rewrite confirmatory manifest](../../results/capacity_length_paired_rewrite_7b_pilot_v1/pilot/eval/confirmatory/eval_manifest_confirmatory_shard_00_of_01.json) 也表明内部 `train[3000:7473]` 已被评测。它们可以继续作为锁定的比较/复现 cohort，但不能再称为 untouched confirmatory set。多个 training seed 能估计训练随机性，却不会把 1,269 道题变成 `1,269 x seeds` 个独立测试样本。

### 6.2 主分析

每个预注册的 teacher/student cell 内：

- 对 short–medium、medium–long、short–long 使用 paired accuracy difference、problem-level bootstrap CI 和 exact/asymptotic McNemar；
- 在该 cell 的三项 primary family 内做 Holm correction；
- seed 级先保存各自效应，再用 seed 和 problem 两层 cluster bootstrap 或预注册的分层模型估计不确定性；
- 同时报告每个 seed，不只报告 seed average。

跨 cell 的主模型可写为：

```text
correct ~ rank * log2(teacher_size) * log2(student_size) + difficulty
```

并为 problem 和 training seed 建立分层/聚类结构。只有 3 个 seed 时，不应过度依赖随机效应方差的精确估计；必要时在最终关键条件增加到 5 seeds。

### 6.3 难度与长度机制

题目难度不能由 formal test 上某个待比较 adapter 的结果定义。可预先使用：

- base student 的冻结 pass/fail 或多次采样 pass rate；
- teacher 正确候选率；
- 数据集已有元数据或题目运算步数代理；
- 与 formal outcomes 隔离的 development split 上训练的难度模型。

建议图形：

- teacher capacity x rank 的 accuracy interaction line plot；
- student capacity 分面的 accuracy–output-token Pareto scatter；
- `K` 对 short/median/long 分位长度的 line plot；
- 按难度分层的 rank effect forest plot；
- equal-example/equal-token 的 paired ablation bars；
- support retention 和筛选难度偏差图。

准确率、长度和合规率必须分别展示。Pareto 图可以辅助解释，但不取代预注册 primary accuracy test。

## 7. 工程组织与证据要求

建议新实验使用独立命名，例如 `capacity_length_ranked_sampling_multiteacher_v1`，不得覆盖当前 `capacity_length_ranked_sampling_7b_v1`。

按项目约定：

- reusable selection、audit 和 analysis logic 放入 `src/length_budget_distill/`；
- runnable entrypoint 使用新的 phase-first `scripts/{phase}_{subphase}_*.py`；
- immutable parent config、training overlay 和 eval overlay 分开存入 `configs/`；
- raw shards、merged pools、datasets、predictions、statistics 和 completion audit 放入新的 `results/` root；
- adapters 放入独立 `checkpoints/` root；
- PNG/PDF 主图放入独立 `figures/` root。

GPU-heavy generation 应按可用 GPU 做一进程一 GPU 的不相交 problem shards，先保存 shard artifacts 再审计合并。每次启动前遵循项目的 `nvidia-smi`、C49 allocation 和 stable-idle-GPU 规则。文档不预先假定某个节点或某块 GPU 可用。

每个阶段完成至少需要：

- config、input、source code 和模型 revision hashes；
- 完整 shard manifests，无 missing/duplicate records；
- rank selection 与 support-retention audit；
- final LoRA 文件及 hash-bound completion marker；
- 每题 prediction 和 aggregate metrics；
- analysis artifact manifest、PNG/PDF 图和独立 completion audit；
- 明确的 `smoke`、`pilot`、`comparative replication` 或 `formal` evidence label。

queued job、部分 shard、只有 adapter 目录或只有汇总 JSON 都不构成实验完成。

## 8. 最小可发表版本与完整版本

最小可发表版本优先保证解释性：

1. 当前 7B -> 1.5B 条件补齐 3 seeds；
2. 一个冻结 `K_max=32` 候选池上的 sampling-density 数据审计；
3. 4 teachers x 3 ranks x 3 seeds 的固定 1.5B-student 主矩阵；
4. random-correct 和 equal-token 两个关键控制；
5. 与 paired rewrite 的机制对照；
6. GSM8K 范围内完整审计与论文图。

完整版本再增加：

- 0.5B/3B students；
- adaptive difficulty routing；
- 5 seeds 的关键端点；
- 经单独审批的 OOD evaluation；
- SFT 与 on-policy shortest-correct 方法的后续比较。

这个顺序的关键是先确认效应可复现、再解释为什么，最后才扩大适用范围。即使多 seed 后准确率排序消失，长度传递、sampling order statistics 和 capacity mismatch 仍然可以形成有效且诚实的研究结论。
