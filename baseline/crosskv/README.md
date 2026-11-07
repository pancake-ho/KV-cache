# CrossKV

[![CI](https://github.com/forlight123/CrossKV/actions/workflows/ci.yml/badge.svg)](https://github.com/forlight123/CrossKV/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

NVIDIA 论文 [*Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse*](https://arxiv.org/abs/2608.03893)
的独立复现。本仓库与 NVIDIA 及论文作者无隶属关系。

本实现覆盖论文的闭式线性方法：让发送模型和接收模型读取同一段历史，将发送模型的
KV cache 映射为接收模型可直接消费的 cache，从而跳过接收模型对历史的完整 prefill。
当前不包含 C2KV，也没有把论文中的可选 MLP 扩展混入主结果。

> Agent 切换方向的新预实验见 `docs/rag_collaboration_handoff.md`：基于 LongBench HotpotQA
> 的 policy-conditioned RAG，以及基于 BFCL 请求的 planner→executor 长历史 handoff，均包含
> front/tail plaintext、Identity、KV reuse 和 late-bound current-request 对照。

默认示例采用论文中的 Qwen3-8B → Qwen3-32B matched-KV 模型对：源/目标分别为
36/64 层，均为 8 个 KV heads、每 head 128 维。实现对应论文方法的三个关键步骤：

1. 从相同校准 token 序列抽取两模型 K/V，并把 K 精确逆 RoPE 到 content space；
2. 对每个目标层，以 matched-head 单源 OLS 的 K/V head-average R² 选择 top-k 源层；
3. 拼接这些源层的全部 KV heads，对每个目标 KV head 拟合带偏置 ridge，推理时重新施加目标 RoPE 并注入 `DynamicCache`。

## 快速开始

需要 Python 3.10+、PyTorch 支持的 GPU，以及足以加载所选模型的显存。安装
[uv](https://docs.astral.sh/uv/) 后执行：

```bash
git clone https://github.com/forlight123/CrossKV.git
cd CrossKV
uv sync --extra test --extra evaluation
uv run pytest -q
```

模型既可填写 Hugging Face ID，也可填写本地 checkpoint 目录。复制配置模板并设置校准数据：

```bash
cp .env.example .env
# 编辑 .env，至少设置 XKV_CALIBRATION_DATA
bash scripts/run_pipeline.sh
```

`XKV_CALIBRATION_DATA` 可指向 Parquet、JSON、JSONL、glob 或包含这些文件的目录；默认文本列为
`text`。脚本依次执行 token packing、源/目标 KV 抽取、top-k 层选择和 ridge mapper 拟合，
产物写入 `XKV_WORK_DIR`。所有模型、设备、attention backend、采样规模和 mapper 超参数都可在
`.env` 中覆盖，完整默认值见 [`.env.example`](.env.example)。

若机器只允许使用已下载的模型，可把模型变量设为本地目录，并设置
`HF_HUB_OFFLINE=1`。不使用 uv 时也可运行
`pip install -e ".[test,evaluation]"`，然后直接调用 `xkv-*` 命令。

论文规模的 Qwen3-8B → Qwen3-32B 运行预计产生约 52 GiB activation store 和约 6 GiB
float32 mapper。模型权重、数据集、activation、mapper 和评测输出均被 `.gitignore` 排除，
需由使用者自行准备；配置文件中的哈希用于核对原始复现实验。

## 当前状态

- RoPE 正逆、ridge、cache shape、token shift 和左填充 batch generation 均有单元测试；
- tiny-Qwen3 恒等 mapper 的映射缓存、continuation logits 和批量生成均与原缓存一致；
- Qwen3-8B → 32B 已完成论文规模 extraction → selection → fit → cache injection；
- 提供 ARC-C、HellaSwag、WinoGrande、MMLU、GSM8K 与 WikiText-2 评测，支持分片、断点续跑和 standalone/transfer 同时计分；
- 正式 mapper 共 1,610,743,808 个参数；64 个 target layer 的校准集平均训练
  `R²_K=0.7492`、`R²_V=0.6379`；
- 本地 evaluator 在同一 300 条有序子集上的 32B standalone 预测与
  `lm-eval 0.4.12` 逐条一致。
- ARC-C、HellaSwag、WinoGrande、MMLU、GSM8K 与 WikiText-2 均已完成全量评测；
  GSM8K 另完成所有 1,024-token 触顶样本的 4,096-token 敏感性重跑。

正式 mapper 在一次 47-token continuation sanity check 上得到 standalone NLL
4.535、transfer NLL 4.723、top-1 agreement 78.7%。这只验证 cache injection，主结果以
完整 HellaSwag validation 为准。

## 开发环境

锁文件记录了已验证的依赖解析。进入仓库后创建环境：

```bash
uv sync --extra test --extra evaluation
source .venv/bin/activate
python -m pytest -q
```

以下详细命令假设已在仓库根目录执行 `source .venv/bin/activate`。项目以 editable 模式安装，
不需要设置 `PYTHONPATH`；也可以在每条命令前使用 `uv run`。

## 论文规模复现：Qwen3-8B → Qwen3-32B

论文配置、checkpoint revision 和数据哈希记录在
`configs/qwen3_8b_to_32b.json`。数据路径只是复现实验记录，可通过 CLI 或 `.env` 替换。

### 1. 固定校准 tokens

```bash
python -m xmodel_kv.cli.prepare \
  --dataset datasets/fineweb-edu/sample/10BT/000_00000.parquet \
  --tokenizer Qwen/Qwen3-8B \
  --output work/qwen3_8b_to_32b/tokens.npy \
  --num-sequences 500 \
  --sequence-length 1024
```

数据加载器会递归识别 Parquet、JSONL 和 JSON，默认读取 `text`；文档以 EOS 分隔后确定性 pack，保证两个模型看到完全相同的 token。

### 2. 分别抽取源/目标 KV

两个模型无需同时驻留 GPU。K 在抽取时已去 RoPE，落盘为 float16；正式 covariance 与求解转为 float32。

```bash
CUDA_VISIBLE_DEVICES=0 python -m xmodel_kv.cli.extract \
  --model Qwen/Qwen3-8B \
  --tokens work/qwen3_8b_to_32b/tokens.npy \
  --output work/qwen3_8b_to_32b/source \
  --role source --stride 4 --device-map auto

CUDA_VISIBLE_DEVICES=1 python -m xmodel_kv.cli.extract \
  --model Qwen/Qwen3-32B \
  --tokens work/qwen3_8b_to_32b/tokens.npy \
  --output work/qwen3_8b_to_32b/target \
  --role target --stride 4 --device-map auto
```

预计 activation store 约 52 GiB：8B 约 19 GiB、32B 约 34 GiB。

### 3. 单源 R² 选 top-12

论文的 layer-selection probe 是 OLS，因此这里 `--ridge 0`；K 与 V 的 R² 先按 head 平均，再等权平均决定每个目标层的源层排序。

```bash
CUDA_VISIBLE_DEVICES=0 python -m xmodel_kv.cli.select \
  --source work/qwen3_8b_to_32b/source \
  --target work/qwen3_8b_to_32b/target \
  --output work/qwen3_8b_to_32b/selection_k12.json \
  --top-k 12 --ridge 0 --device cuda:0
```

### 4. 拟合 production ridge

```bash
CUDA_VISIBLE_DEVICES=0 python -m xmodel_kv.cli.fit \
  --source work/qwen3_8b_to_32b/source \
  --target work/qwen3_8b_to_32b/target \
  --selection work/qwen3_8b_to_32b/selection_k12.json \
  --output work/qwen3_8b_to_32b/mapper_k12 \
  --ridge 0.01 --device cuda:0 --weight-dtype float32
```

每个目标层输出一个 safetensors 文件，可断点续跑。正式 mapper 为 1.61B 参数，float32 约 6 GiB，与论文 Table 12 一致。求解目标为平均 MSE 加 `λ‖W‖²`，即使用归一化 covariance `XᵀX/N`；这与论文中 λ=1 会明显退化的消融量纲一致。校准 observations 少于 feature dimension 的 smoke 会自动使用等价 dual solve；论文规模使用 primal solve。

### 5. 单段 continuation 验证

`xkv-verify` 同时计算目标模型正常 prefill 和 mapped-cache 两条路径。映射缓存故意截止在 prefix 最后一个 token 之前，让目标亲自处理该 token，从而产生第一个 continuation token 所需的目标 logits。

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m xmodel_kv.cli.verify \
  --source-model Qwen/Qwen3-8B \
  --target-model Qwen/Qwen3-32B \
  --mapper work/qwen3_8b_to_32b/mapper_k12 \
  --source-device cuda:0 --target-device cuda:1 \
  --prefix-length 128 \
  --text 'A sufficiently long prefix and continuation for a likelihood sanity check ...'
```

### 6. HellaSwag 主结果

将 validation split 单独传给 `--dataset`。mapper 会预先常驻 target GPU；每条结果立即写 JSONL，重复执行会跳过已完成 index。

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m xmodel_kv.cli.hellaswag \
  --dataset datasets/hellaswag/validation-00000-of-00001.parquet \
  --source-model Qwen/Qwen3-8B \
  --target-model Qwen/Qwen3-32B \
  --mapper work/qwen3_8b_to_32b/mapper_k12 \
  --source-device cuda:0 --target-device cuda:1 \
  --output outputs/hellaswag/qwen3_8b_to_32b.jsonl
```

论文的复现目标是 target standalone `acc_norm=82.65`、transfer
`acc_norm=78.67`、retention `95.2%`。HellaSwag adapter 严格使用官方
`activity_label: context` 输入、单空格 target delimiter，并按清理后候选的字符数计算
`acc_norm`（不是 token 数）。可用 `--num-shards N --shard-index I` 分片；不同 shard
使用不同输出文件。完成后用以下命令检查 10,042 个 index 完整且无重复并合并指标：

```bash
xkv-summarize-hellaswag \
  outputs/hellaswag/qwen3_8b_to_32b_protocol_v2_shard*.jsonl \
  --expected-documents 10042 \
  --output outputs/hellaswag/qwen3_8b_to_32b_protocol_v2_summary.json
```

本次完整运行结果如下。standalone 与论文相差 `-0.05 pp`，transfer 相差
`+0.56 pp`，复现出的 retention 高 `+0.71 pp`：

| HellaSwag `acc_norm` | 论文 | 本复现 | 差值 |
|---|---:|---:|---:|
| 32B standalone | 82.65% | 82.60% (8295/10042) | -0.05 pp |
| 8B→32B transfer | 78.67% | 79.23% (7956/10042) | +0.56 pp |
| retention | 95.20% | 95.91% | +0.71 pp |

严格合并结果保存在
`outputs/hellaswag/qwen3_8b_to_32b_protocol_v2_summary.json`；三个逐样本 JSONL
保留原始/归一化 log-likelihood、预测、label、字符数和 token 数，可用于误差分析。

## 完整 benchmark 数据与 Table 13

原始复现实验在本地 `datasets/` 中准备了 ARC-Challenge、WinoGrande、MMLU、GSM8K、
WikiText-2 和 CoQA；这些数据不随仓库分发。无需下载 5.4 TB 的 FineWeb-Edu 全集：校准只使用
论文要求的 500×1024 tokens，一份容量足够的文本子集即可。
所有评测数据的 Hugging Face revision、实际 split、行数和 SHA256 固定在
`configs/evaluation_datasets.json`。

除 HellaSwag 外，其余任务统一用 `xkv-benchmark`。例如 ARC-C 单卡运行：

```bash
CUDA_VISIBLE_DEVICES=0 xkv-benchmark \
  --task arc_challenge --dataset datasets/arc_challenge \
  --source-model Qwen/Qwen3-8B \
  --target-model Qwen/Qwen3-32B \
  --mapper work/qwen3_8b_to_32b/mapper_k12 \
  --source-device cuda:0 --target-device cuda:0 \
  --output outputs/table1/qwen3_8b_to_32b_arc_challenge.jsonl
```

任务名可替换为 `winogrande`、`mmlu`、`gsm8k` 或 `wikitext2`。全量运行使用
`--num-shards 3 --shard-index 0/1/2` 和不同输出文件；合并时用
`xkv-summarize-benchmark --task TASK shard*.jsonl --expected-documents N` 验证 index
覆盖。协议分别是 ARC-C zero-shot `acc_norm`、WinoGrande zero-shot `acc`、MMLU
first-5-shot `acc`、GSM8K 8-shot CoT flexible exact match，以及 WikiText-2
non-overlapping 2048-token chunk 的 1024-token mapped prefix 条件 PPL。前三个任务的
prompt/request 已逐条与 `lm-eval 0.4.12` 的 task object 对照。

当前完整结果如下；百分数差值均为本复现减论文。GSM8K 主表采用 4,096-token
长尾替换结果，原因和 1,024-token 对照见表后说明。

| 指标 | 论文 standalone | 本复现 standalone | 论文 transfer | 本复现 transfer | 论文 retention | 本复现 retention |
|---|---:|---:|---:|---:|---:|---:|
| ARC-C `acc_norm` | 61.01% | 61.26% | 57.34% | 56.40% | 94.0% | 92.06% |
| HellaSwag `acc_norm` | 82.65% | 82.60% | 78.67% | 79.23% | 95.2% | 95.91% |
| WinoGrande `acc` | 70.01% | 72.77% | 63.69% | 67.25% | 91.0% | 92.41% |
| MMLU 5-shot `acc` | 82.17% | 82.09% | 72.74% | 72.90% | 88.5% | 88.80% |
| GSM8K 8-shot `acc` | 95.15% | 94.01% | 65.50% | 69.22% | 68.8% | 73.63% |
| WikiText-2 PPL | 6.79 | 6.7803 | 7.98 | 7.2591 | — | — |

ARC-C、MMLU 和 WikiText-2 的 target baseline 与论文分别只差 +0.25 pp、−0.08 pp
和 −0.01 PPL。WinoGrande 的 prompt 与官方 harness 一致，但本地 checkpoint/runtime
的 standalone 比论文高 2.76 pp，因此同时保留绝对 accuracy 和 retention，不用基线差异
掩盖结果。严格合并 summary 和逐样本 JSONL 的本地运行路径为 `outputs/table1/`，大体积结果
不随仓库分发。

论文只说明 GSM8K 使用 8-shot CoT 和 greedy `acc`，没有披露最大生成长度。本地 Qwen3
经常在 256 tokens 后仍未给最终答案，因此正式运行采用 lm-eval 的 flexible number
extraction，并在出现完整 final-answer 句式、EOS、`Q:` 或 `<|im_end|>` 时逐样本停止。
统一 1,024-token 全量结果为 standalone `92.65%`、transfer `66.72%`、retention
`72.01%`；共有 172 个 index 的任一侧触顶。只将这些 index 以 4,096 tokens 重跑并替换
后，得到主表的 `94.01% / 69.22% / 73.63%`。长预算净恢复 standalone 18 题、transfer
33 题，但仍无法同时对齐论文的 `95.15% / 65.50%`，说明论文 checkpoint/runtime 或
未披露生成配置存在差异，不能通过选择更接近论文的单侧预算来消除。两份严格 summary
分别是 `qwen3_8b_to_32b_gsm8k_max1024_summary.json` 和
`qwen3_8b_to_32b_gsm8k_max4096_retry_summary.json`。

## CoQA 多轮 handoff（Qwen3 14B↔32B）

Qwen3-14B 到位后，已复用同一批 FineWeb-Edu 校准 tokens 完成论文 Figure 4 的两个
production mapper：14B→32B 使用 `k=8`（1,073,872,896 参数，float32 约 4.00 GiB），
32B→14B 使用 `k=20`（1,677,803,520 参数，约 6.25 GiB）。校准集平均训练 R² 分别为
`K/V=0.7647/0.6647` 与 `0.8313/0.7488`。完整配置和本地模型文件哈希记录在
`configs/qwen3_14b_32b_coqa.json`。

评测使用 `EleutherAI/coqa` validation，因为它保留官方 F1 所需的三组额外人工答案；
原先的 `stanfordnlp/coqa` Parquet 内容相同，但只有单参考答案。论文只说从五个域取
100 段对话，没有公开 index 或 seed，因此本复现明确采用每域按 Parquet 顺序前 20 段，
并在 turn depth 1、3、5、7、10 各做一次 source-cache→target handoff。历史使用 primary
gold answer，prompt、首行答案抽取和多参考 leave-one-out F1 与 lm-eval 0.4.12 CoQA
adapter 一致。

```bash
CUDA_VISIBLE_DEVICES=0 xkv-coqa \
  --dataset datasets/coqa_eleutherai/coqa_validation.parquet \
  --source-model Qwen/Qwen3-14B \
  --target-model Qwen/Qwen3-32B \
  --mapper work/qwen3_14b_32b/mapper_14b_to_32b_k8 \
  --source-device cuda:0 --target-device cuda:0 \
  --conversations-per-domain 20 --turns 1,3,5,7,10 \
  --generation-batch-size 16 --max-new-tokens 256 \
  --output outputs/coqa/qwen3_14b_to_32b_k8_first20_per_domain.jsonl
```

反向替换模型路径和 `mapper_32b_to_14b_k20` 即可。两个方向均通过严格覆盖检查，各有
`100×5=500` 条且无重复：

| 方向 / turn depth | 1 | 3 | 5 | 7 | 10 |
|---|---:|---:|---:|---:|---:|
| 32B standalone F1 | 42.92 | 75.87 | 79.28 | 76.06 | 79.11 |
| 14B→32B F1 | 58.33 | 74.58 | 78.12 | 71.26 | 81.76 |
| drift（pp） | -15.42 | 1.29 | 1.16 | 4.79 | -2.65 |
| 14B standalone F1 | 58.86 | 72.20 | 76.50 | 70.45 | 80.62 |
| 32B→14B F1 | 49.70 | 68.70 | 72.95 | 67.20 | 76.90 |
| drift（pp） | 9.16 | 3.50 | 3.55 | 3.25 | 3.73 |

公开协议下 turn-1 是显著异常点：没有历史短答案示例时，post-trained Qwen3 completion
会生成解释性长答案，standalone 的 256-token 触顶数为 32B 16/100、14B 48/100。
从 turn 3 起，前序 gold answers 自然提供了短答案格式；四个后续深度合并后，14B→32B
为 `77.58/76.43`（standalone/transfer，drift 1.15 pp），32B→14B 为
`74.94/71.44`（drift 3.51 pp）。论文图中的 standalone 从 turn 1 就约为 0.87–0.90，
但没有披露其 100 段抽样、生成上限，或是否额外加了短答案指令/初始历史。因此这里不通过
事后调 prompt 追图；逐条 generation、F1 和 token-cap 均保留在 `outputs/coqa/`，便于后续
在获得作者配置后直接重跑。严格 summary 是两个 `*_summary.json` 文件。

## Agent policy imprinting 预实验

同模型 agent-switch 的受控 cache-surgery runner 已加入 `xkv-policy-imprinting`。它把
target-native、source-native、identity stitch、source-history→target-prefix hybrid 和
target suffix replay 放在同一个 token-exact 协议下，并记录 tool/argument EM、action KL、
source-action leakage 与逐层 H-K/V 距离。实验假设、因果控制、Go/No-Go 标准和当前 pilot
结果见 [`docs/policy_imprinting_preexperiment.md`](docs/policy_imprinting_preexperiment.md)；
BFCL 自然历史、Qwen3-8B/14B、causal patching 与 replay curve 的阶段结论见
[`docs/policy_imprinting_findings.md`](docs/policy_imprinting_findings.md)。

真实任务级 handoff 已加入 `xkv-hotpot-agents`：Agent A 读取 LongBench HotpotQA 的完整问题与
passages 并生成证据报告，Agent B 接收 `A 输入 + A 输出` 后给短答案，分别比较 B 从头 prefill、
identity stitch 与直接复用 A-history KV。200 题最终准确率、成对置信区间、具体翻转样例和当前
边界见 [`docs/hotpot_agent_handoff.md`](docs/hotpot_agent_handoff.md)。

进一步的非对称 `hop1→final` 协议让 A 只输出桥接实体/第一跳证据，B 完成第二跳；已在
standard-200 和 E-300 上运行，并按重复 question 去重为 pooled-490。分 split 准确率、长历史
探索性结果和数据去重审计见
[`docs/hotpot_hop1_handoff.md`](docs/hotpot_hop1_handoff.md)。

## 与论文的对应关系和已知边界

| 项目 | 论文 | 本实现 |
|---|---:|---:|
| 模型对 | Qwen3 8B→32B | `Qwen/Qwen3-8B` → `Qwen/Qwen3-32B` |
| 校准 | FineWeb-Edu 500×1024 | 已完成，stride 4，共 128,000 observations |
| token sampling | stride 4，约 128K | 相同 |
| layer selection | single-source head-average R² | 相同，K/V 等权 |
| production map | per-target-head ridge, λ=0.01 | 相同 |
| source features | top-k layers × all source KV heads | 相同 |
| K | source inverse RoPE → map → target RoPE | 相同 |
| forward/covariance | bf16 / fp32 | 相同 |
| benchmark | lm-eval completion mode | 五个 accuracy/PPL adapter 已实现；分类任务 prompt 与 lm-eval 0.4.12 对齐 |
| latency | 8×H100 + FlashAttention2 | 尚未实现论文 latency harness |

论文原始实现尚未开源，以下细节只能按正文消歧：source-layer probe 按论文 Eq. (2) 使用
matched head，production mapper 则按 §3.1/§3.2 拼接所选层的全部 source heads，允许
cross-head 信息流。实现把这两个阶段明确分开，便于后续做消融。

## 开源说明

- `src/xmodel_kv/` 是可安装的 Python 包，`tests/` 包含无需下载大模型即可执行的单元测试；
- 依赖外部模型或数据的集成测试会在对应 `XKV_*` 环境变量未设置时跳过；
- 历史实验协议和结果说明保留在 `docs/`，大体积原始输出不随仓库分发；
- 欢迎通过 [Issues](https://github.com/forlight123/CrossKV/issues) 报告可复现性问题。

本项目采用 [Apache License 2.0](LICENSE)。
