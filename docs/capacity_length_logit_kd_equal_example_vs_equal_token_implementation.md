# Equal-example 与 Equal-token Logit-KD 实现对照

## 1. 文档范围

本文说明以下两个 seed-17、GSM8K、Qwen2.5-7B 到 Qwen2.5-1.5B 在线
logit-level knowledge distillation 实验的实现差异：

- Equal-example：`capacity_length_logit_kd_seed17_v1`
- Equal-token：`capacity_length_logit_kd_equal_token_seed17_v1`

两者不是两种 KD 损失。它们共用同一套在线 teacher-student 前向、同一个
completion-only CE+KL 损失和同一套训练入口。核心区别是训练 JSONL 的抽样方式，
以及随之变化的样本数、监督 token 数、optimizer steps、匹配的 SFT 基线和独立选择出的
KD 超参数。

这里的 `equal-example` 和 `equal-token` 先在父级 SFT 数据构造阶段定义，再由两个
logit-KD 协议分别读取对应的不可变 JSONL。它们不表示在 KD 训练循环中动态重采样。

## 2. 共同的上游 teacher traces

两个协议都只使用 Qwen2.5-7B-Instruct 的三种轨迹条件：

- `short_128`：teacher prompt 要求解答不超过 128 solution tokens；
- `medium_256`：teacher prompt 要求不超过 256 solution tokens；
- `long_512`：teacher prompt 要求不超过 512 solution tokens。

每个题目和条件生成三个候选。父级流水线选择答案正确、满足预算的候选，并在所有
12 个 capacity-by-length 条件上取共同题目交集。seed-17 正式数据中，7B 的三个长度条件
均有同一组 881 个问题，每个问题在每个条件下各保留一条 teacher completion。

Teacher 生成时看得到长度预算，但 student 训练和评测提示词只有：

```text
Problem:
<question>

Solve the problem and end with a line in the form: Answer: <final answer>.
```

因此，student 在推理时没有收到 `128/256/512` 或长度指令。长度条件通过三个独立
LoRA adapter 的训练 completion 分布体现，而不是一个可在推理时切换的显式控制变量。

## 3. 数据构造差异

### 3.1 Equal-example

Equal-example 对每个长度条件保留完整的 881 题共同交集。因此三个 adapter 看到相同的
problem IDs 和相同的样本数，但 completion 长度不同：

| 条件 | 记录数 | 原始 solution tokens | 实际 KD loss positions | Optimizer steps |
|---|---:|---:|---:|---:|
| short-128 | 881 | 31,443 | 33,205 | 221 |
| medium-256 | 881 | 74,339 | 76,101 | 221 |
| long-512 | 881 | 156,656 | 158,418 | 221 |

它控制的是题目数和 optimizer steps，不控制监督 token 数。long-512 的原始监督 token
约为 short-128 的 4.98 倍。

### 3.2 Equal-token

父级数据构造首先计算所有 capacity-by-length 条件的完整 solution-token 总数，并取最小值
31,443 作为目标。对每个条件和训练 seed，代码使用
`canonical_sha256([config_hash, condition_slug, seed])` 导出确定性 shuffle seed，随后：

1. 打乱该条件下的完整 traces；
2. 依次加入整条 trace，但绝不让累计 token 超过目标；
3. 扫描结束后按 `problem_id` 排序并写入 JSONL。

实现位于 `src/length_budget_distill/factorial.py::deterministic_equal_token_subset`，入口为
`scripts/5_3_build_capacity_length_sft_data.py`。因为不切分单条 completion，所以 token 总数
只能近似相等：

| 条件 | 记录数 | 原始 solution tokens | 实际 KD loss positions | Optimizer steps |
|---|---:|---:|---:|---:|
| short-128 | 881 | 31,443 | 33,205 | 221 |
| medium-256 | 381 | 31,442 | 32,204 | 96 |
| long-512 | 179 | 31,407 | 31,765 | 45 |

`实际 KD loss positions` 来自正式 adapter 的 `training_metrics.json`。它比
`metadata.solution_token_count` 多两个 assistant 结束模板 token/样本。因此，equal-token
严格控制的是注册的原始 solution-token 总数，实际进入 CE/KL 的 token positions 仍为
33,205、32,204 和 31,765，最大相差 1,440。

Equal-token 也不控制 optimizer steps。训练循环每次处理一个样本并累积四个样本，使用
`ceil(record_count / 4)` 个 optimizer steps，所以三个条件分别为 221、96 和 45 steps。

### 3.3 数据集合的包含关系

| 条件 | Equal-token 相对 Equal-example 的关系 |
|---|---|
| short-128 | 两个 JSONL 字节级相同，SHA256 均为 `4328598f...22bba` |
| medium-256 | Equal-token 的 381 条全部是 881 条 Equal-example 数据的子集 |
| long-512 | Equal-token 的 179 条全部是 881 条 Equal-example 数据的子集 |

因此，short-128 两个实验的数据本身没有差别；其最终 KD adapter 不同，是因为两个协议
独立选择了不同的 `alpha`。medium-256 和 long-512 同时改变了数据量和最终超参数。

## 4. 两个协议共用的在线 Logit-KD 实现

### 4.1 模型和参数化

- Teacher：`Qwen/Qwen2.5-7B-Instruct`，revision
  `a09a35458c702b33eeacc393d103063234e8bc28`，BF16，冻结并保持 eval mode；
- Student：`Qwen/Qwen2.5-1.5B-Instruct`，revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，BF16；
- Student LoRA：rank 4、alpha 16、dropout 0.05、`all-linear`；
- Teacher/student 共用并校验同一个 151,665-token 有效词表；
- 训练 seed：17。

### 4.2 Completion-only token 对齐

`src/length_budget_distill/logit_kd.py::tokenize_completion_record` 分别构造：

1. 只有 user 消息、带 generation prompt 的 token IDs；
2. user+assistant 完整对话的 token IDs。

代码要求前者是后者的精确前缀。loss 只作用于 assistant completion 和 assistant 结束模板
tokens，user prompt 不进入 CE 或 KL。最大完整序列长度为 2,048；超长样本直接报错，
不静默截断。

### 4.3 精确 KD 损失

Teacher 和 student 在同一个 teacher-forced completion 上前向。Teacher 不计算梯度，
student 只更新 LoRA 参数。每个 completion token 的损失为：

```text
L = (1 - alpha) * CE(target, student)
    + alpha * T^2 * KL(teacher_T || student_T)
```

其中 `KL(teacher_T || student_T)` 是 teacher-to-student forward KL，并在完整的 151,665 个
有效 vocabulary IDs 上精确计算。配置中的 `top_k=64` 只用于保存 matched-logit 分析快照，
不近似训练时的 KL。

### 4.4 优化设置

两个协议完全相同：

- 1 epoch；
- 每次前向 1 个样本，gradient accumulation 4；
- AdamW，learning rate `2e-5`，weight decay 0；
- linear scheduler，warmup ratio 0.03；
- gradient clipping 1.0；
- gradient checkpointing；
- 每个 accumulation group 按该 group 的实际样本数归一化，因此最后一个 partial group
  也会正常更新。

`hybrid_kd_loss` 先在单个样本的 completion tokens 上取均值，训练循环再对一个
accumulation group 中的样本取均值。因此梯度层面是“样本等权”，不是把整个数据集的
所有 token 拼接后做一次全局 token-weighted mean。Equal-token 控制 token 总量，并不使
每个 token 在两个协议之间获得完全相同的总体权重。

Equal-example 三个条件均为 221 steps、7 warmup steps。Equal-token 的 short/medium/long
分别为 221/96/45 steps，warmup 为 7/3/2 steps。两种设计都不是 equal-compute 设计。

## 5. 超参数选择不是共享的

每个协议都独立训练同一套 27 个 validation adapters：

- `alpha`：0.25、0.5、0.75；
- temperature：1、2、4；
- 三个长度条件；
- 验证集：GSM8K `train[2000:2500]`，500 题；
- 每个协议最终只选择一组跨三个预算共享的 `(alpha, temperature)`。

选择规则依次为：

1. 如果存在三个预算的 compliance 都不低于匹配 SFT 的候选，只在这些候选中选择；
2. 最大化三个预算中最小的 accuracy delta；
3. 最大化 macro accuracy delta；
4. 最小化平均训练 KL；
5. 优先更低的 alpha，再优先更低的 temperature。

两个实验都没有找到在三个预算上保持 SFT compliance 的候选，因此都使用第 2 项开始的
fallback 排序。最终选择不同：

| 协议 | Selected alpha | Selected temperature |
|---|---:|---:|
| Equal-example | 0.75 | 2 |
| Equal-token | 0.50 | 2 |

因此，当前正式结果同时包含“数据平衡方式改变”和“独立模型选择结果改变”。它不是一个
只替换训练子集、同时固定 `alpha/T` 的纯单因素比较。

## 6. 正式评测实现

- 正式 cohort：GSM8K `test[50:1319]`，每个 adapter 1,269 题；
- 解码：greedy，temperature 0，top-p 1；
- 所有条件统一允许最多生成 512 tokens；
- budget compliance 在分析时按实际输出是否不超过 128/256/512 tokens 计算；
- Equal-example KD 与对应的 Equal-example SFT adapter 比较；
- Equal-token KD 与对应的 Equal-token SFT adapter 比较。

评测提示词没有显式预算，且三个条件都允许生成 512 tokens。因此 compliance 衡量的是
adapter 是否从训练分布中学会短输出，而不是 decoder 强制截断后的合规率。

正式 KD 结果如下：

| 条件 | Equal-example accuracy | Equal-token accuracy | Equal-example compliance | Equal-token compliance |
|---|---:|---:|---:|---:|
| short-128 | 71.00% | 68.95% | 1.65% | 7.96% |
| medium-256 | 70.84% | 68.95% | 57.60% | 58.47% |
| long-512 | 72.03% | 69.58% | 100.00% | 100.00% |

两个结果的定性形状相似：KD 把三个条件的准确率拉到约 69%--72%，但 short/medium 的
长度控制明显退化。数值、adapter、协议哈希和预测文件均不同，不能当作同一实验。

## 7. 配置、代码与正式产物

### 7.1 配置与协议

- Equal-example 配置：`configs/capacity_length_logit_kd_seed17_v1.json`
- Equal-token 配置：`configs/capacity_length_logit_kd_equal_token_seed17_v1.json`
- 通用 KD 协议：`docs/capacity_length_logit_kd_protocol.md`
- Equal-token 补充协议：`docs/capacity_length_logit_kd_equal_token_protocol.md`

Equal-example v1 配置没有显式 `supervision.mode`；
`src/length_budget_distill/logit_kd.py::supervision_mode` 为向后兼容将其默认解释为
`equal_example`。Equal-token 配置显式注册 `supervision.mode=equal_token`、token 定义、
目标和容差。

### 7.2 关键实现

- 数据构造：`scripts/5_3_build_capacity_length_sft_data.py`
- 确定性 equal-token 子集：
  `src/length_budget_distill/factorial.py::deterministic_equal_token_subset`
- 通用 KD 逻辑：`src/length_budget_distill/logit_kd.py`
- 单 adapter 训练入口：`scripts/9_1_train_logit_kd.py`
- 验证评测和选择：`scripts/10_1_eval_logit_kd.py`、
  `scripts/10_2_select_logit_kd_hparams.py`
- 正式分析：`scripts/11_2_analyze_logit_kd_experiment.py`
- 独立完成审计：`scripts/12_1_audit_logit_kd_completion.py`

### 7.3 独立结果根目录

- Equal-example：`results/capacity_length_logit_kd_seed17_v1/`
- Equal-token：`results/capacity_length_logit_kd_equal_token_seed17_v1/`

两个根目录都已有通过审计的 `FORMAL_COMPLETE`。正式报告分别位于：

- `results/capacity_length_logit_kd_seed17_v1/formal/analysis/experiment_report.md`
- `results/capacity_length_logit_kd_equal_token_seed17_v1/formal/analysis/experiment_report.md`

## 8. 复现实验

Equal-example：

```bash
CONFIG=configs/capacity_length_logit_kd_seed17_v1.json \
  DRY_RUN=1 bash scripts/9_0_submit_logit_kd_experiment.sh

CONFIG=configs/capacity_length_logit_kd_seed17_v1.json \
  bash scripts/9_0_submit_logit_kd_experiment.sh
```

Equal-token：

```bash
CONFIG=configs/capacity_length_logit_kd_equal_token_seed17_v1.json \
  DRY_RUN=1 bash scripts/9_0_submit_logit_kd_experiment.sh

CONFIG=configs/capacity_length_logit_kd_equal_token_seed17_v1.json \
  bash scripts/9_0_submit_logit_kd_experiment.sh
```

上述命令会运行 parent preflight、GPU smoke、27-cell validation sweep、独立超参数选择、
3 个正式 adapter、正式评测、matched-logit 抽取、分析和最终审计。已有完整结果根目录时，
脚本会通过完成标记跳过或拒绝覆盖相应产物。

## 9. 解释边界

1. Equal-example 回答“相同题目数和 steps 下，不同 teacher trace 长度分布有什么结果”；
   它不控制监督 token 数。
2. 当前 Equal-token 回答“原始 teacher solution-token 总量近似相同时有什么结果”；它不控制
   题目数、optimizer steps、warmup/scheduler exposure 或实际 chat-template loss positions。
3. 两个协议独立选择 `alpha/T`，所以它们不是严格的单因素 causal comparison。
4. 长度条件通过独立 adapter 表示，推理 prompt 不含显式预算；当前实验不能证明一个模型
   能在运行时按给定 token budget 切换输出长度。
5. 结果只覆盖 GSM8K 和训练 seed 17，不估计 training-seed variability，也不支持 OOD 结论。
6. 正式 `test[50:1319]` 已被父级实验观察过；这是锁定的比较性重跑，不是新的 untouched test。
