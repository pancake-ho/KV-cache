# Agent-switch KV policy imprinting：预实验结论与瓶颈

本文档记录 2026-08-10 前完成的同模型 agent-switch 预实验。目标不是估计真实部署中的
最终发生率，而是回答两个问题：policy imprinting 是否确实存在，以及错误状态主要位于哪里。

## 结论先行

现象成立，但命题必须收窄：**直接复用 source agent 形成的历史 KV，可能使 target 的动作
系统性朝 source policy 偏移；当 target policy 不足以覆盖这种偏移时，会直接生成 source
动作。它不是所有 agent handoff 都必然发生的错误。**

目前已经同时满足以下因果证据：

1. source/target 使用同一个模型，历史和 assistant readout 的 token ID 完全相同；
2. identity cache surgery 保持 target 行为，排除了切片、拼接和 re-RoPE 实现错误；
3. 零 replay 的 hybrid 在合成任务和 BFCL 自然历史中都出现 `B = S != A`；
4. source-action margin 在严格自然样本中系统性上升；
5. 全量 target replay 恢复 A；
6. causal patching 能用一个连续中层区间双向控制 `A↔S`。

2026-08-10 新增的三向 policy 对照进一步排除了“stitching 只造成无方向退化”的竞争解释：
matched-source 的 hybrid 全部保持 target，而把 source 改成一个与 target、常见位置先验都不同的
third policy 后，hybrid 全部执行 third policy。这个结果在位置型和非位置内容型 policy、等长
prefix 和需要 re-RoPE 的 shifted prefix 上均成立。

因此，NVIDIA/MoT 式“把同一上下文的 KV 换坐标”并没有构造 target policy 下的反事实历史
状态。对 agent switch 来说，坐标正确并不等于语义状态正确。

## 评测口径

每个 case 都包含四个主实验臂：

- A：target prompt/tool schema 下原生 prefill；
- S：source prompt/tool schema 下原生 prefill；
- I：从 A 自己切出历史 KV 再接回 target prefix；
- B：把 S 的历史 KV re-RoPE 后接到 target-native prefix。

早期冲突实验的严格有效集合要求 A、S 都命中各自预先指定的完整 tool call，且 `A != S`。
三向实验改用 arm-specific gating：matched 要求 A/S 都执行同一目标 policy，third 要求 A/S
分别执行 target/third policy，null 只要求 S 是合法 9-way action。只在对应 eligible 集合上报告
主判据，同时保留不依赖 ground truth 的 B-vs-A native agreement。这样不会把模型原本不会做
selection 的错误算成 KV 错误。

BFCL 自然实验使用官方 `memory_prereq_conversation` 文本和官方
`archival_memory_add(text)` schema。`tagged` 版本只在每条原始记录前增加随机不透明 ID，
要求把所选 ID 作为短参数；`verbatim` 版本不增加 ID，要求传递所选自然记录原文。历史最后
增加一个各臂完全相同的 batch-complete trigger。数据仓库固定在 BFCL leaderboard 指定的
commit `f7cf7359b7ac615a0b294831c5ba2bc95ee4a000`。

## 行为结果

### 决定性三向 policy 对照（Qwen3-8B，seed 2027）

每个方向生成 25 份独立随机 history，每份包含 9 个可信 `choose_record(record_id)` 动作，均匀
随机猜测基线为 1/9=11.1%。总计 600 次运行来自
`2 families × 2 directions × 3 source arms × 2 layouts × 25 histories`；统计上的基本
样本量是每个方向 25 份 history，不能把 600 次相关运行当成 600 个独立样本。

- `matched`：source 与 target 语义 policy 相同，只改变措辞；判据为 `B=A`。
- `third`：source 使用与 target 不同的第三策略；判据为 `B=third=S` 且 `B!=A`。
- `null`：source 不给 selection policy，用于测量模型/拼接的默认先验。
- `clean`：四种 source/target private prefix 被惰性标签精确补到相同 token 长度，实测
  `actual_prefix_shift=0`，因此 re-RoPE 是 no-op。
- `shifted`：仅给 source 增加惰性标签，实测每例 `actual_prefix_shift=+74`，实际执行 re-RoPE。

主判据结果如下。括号内为 Wilson 95% CI；content 中 2 个实例的模型原生 cheapest 判断错误，
按预注册 eligibility 规则剔除，而不是把基础任务错误算成 KV transfer 结果。

| family | target direction | arm / 判据 | clean | shifted |
|---|---|---|---:|---:|
| position | first | matched: `B=A` | 25/25 (86.7--100%) | 25/25 (86.7--100%) |
| position | last | matched: `B=A` | 25/25 (86.7--100%) | 25/25 (86.7--100%) |
| position | first | third=middle: `B=third` | 25/25 (86.7--100%) | 25/25 (86.7--100%) |
| position | last | third=middle: `B=third` | 25/25 (86.7--100%) | 25/25 (86.7--100%) |
| content | urgent | matched: `B=A` | 25/25 (86.7--100%) | 25/25 (86.7--100%) |
| content | cheapest | matched: `B=A` | 23/23 (85.7--100%) | 23/23 (85.7--100%) |
| content | urgent, source=cheapest | third: `B=third` | 23/23 (85.7--100%) | 23/23 (85.7--100%) |
| content | cheapest, source=urgent | third: `B=third` | 23/23 (85.7--100%) | 23/23 (85.7--100%) |

内容型任务中 urgent 和 cheapest 记录的位置分别独立随机化且互不相同；25 份 history 中两个
目标都覆盖了 0--8 的全部位置。因此上表的 23/23 不是 first/last 操作或固定证据位置的产物。

null arm 也不支持“hybrid 退化后固定输出 first/middle”的解释：位置型四个 direction/layout
格子中 `B=A` 均为 25/25，而 `B=middle` 均为 0/25。内容型 clean 下 `B=A` 为 36/48，
`B=third` 仅 8/48；相比之下，明确写入 third source policy 后是 46/46。也就是说，第三动作
不是无 policy 时自然出现的通用 fallback，而是由 source 的具体 policy 定向决定。

同一 history 上的 paired exact McNemar 检验也已写入结构化汇总。third 相对 null 的
`B=third` 提升为：position 两个方向在 clean/shifted 均 +100 pp（每格 `p=5.96e-8`）；
content target-cheapest 为 +91.3 pp（`p=9.54e-7`），target-urgent 为 +73.9 pp clean
（`p=1.53e-5`）和 +82.6 pp shifted（`p=3.81e-6`）。matched 相对 third 的 `B=A`
对照同样全部显著（每格 `p<=2.38e-7`）。因此结果不依赖把模型默认动作假设成均匀随机；
它直接比较同一份 history 在是否写入具体 source policy 时的配对变化。

实现完整性对照同样全部通过：600/600 identity stitch 与 A 动作一致，最大 identity mean-KL
为 0；600/600 full target replay 与 A 动作一致，最大 mean-KL 为
`7.09e-7`；所有 600 个 zero-replay 输出都是合法的 9-way action。clean 与 shifted 必须继续
分表报告，当前没有把两者合并成一个比例。

这组结果把结论从“B 与 A 不同”提升为更强的因果陈述：**在 model、target prefix、共享
history、动作集合和 cache surgery 全部固定时，只改变形成 history KV 的 source policy，
就能把 target readout 定向切换到 source 指定的 matched/third 策略。** 表示退化、首因先验和
纯 re-RoPE 伪影无法同时解释这些对照。

Qwen3-14B 另做了 seed 41 的跨规模 smoke test：clean/shifted 中 matched 均为 4/4 `B=A`，
third 均为 4/4 `B=third`，shifted 的实测位移同样为 +74。它只有每方向一个 history，只作为
代码路径和跨规模方向一致性的 pilot，不进入 n=25 主统计或置信区间。

### 把 target policy 放在历史之后（Qwen3-8B，seed 2027）

为测试 late binding，另构造了不改变历史 KV 的接收端干预。令 $H_0$ 为 9 条记录及其
acknowledgement，不包含最终 selection query；$Q_t$ 是放在最后一条 user message 中的 target
policy。主要比较：

- front：`target-prefix + KV_source(H0 + generic-query)`，即原始 B；
- tail-native：`neutral-receiver-prefix + H0 + Q_t` 从头计算，作为 late-policy 的 A；
- tail-stitched：`neutral-receiver-prefix + KV_source(H0) + native(Q_t)`；
- keep-source-tail：保留完整 `source-prefix + KV_source(H0)`，直接追加并原生计算 $Q_t$。

plain-tail 的 $Q_t$ 只是在原 target policy 前增加 `The batch is complete.`，不含额外的
“override earlier rule”措辞。每方向仍使用同样的 25 份独立 history、9-way 动作，并分别报告
shift=0 clean 和 shift=+74 shifted：

| family / target | eligible | front `B=S` | tail-stitched `B=A` | keep-source-tail `B=A` |
|---|---:|---:|---:|---:|
| position / first | 25 | 25/25 | 25/25 | 25/25 |
| position / last | 25 | 25/25 | 25/25 | 25/25 |
| content / urgent | 23 | 23/23 | 23/23 | 23/23 |
| content / cheapest | 23 | 23/23 | 23/23 | 23/23 |

上表在 clean、shifted 中完全相同；tail-vs-front 的 paired exact McNemar 每格
`p<=2.38e-7`。因此，在当前短受控 selection 任务中，**把完整 target policy 作为生成前的最后
一条指令，足以在行为层覆盖 source-imprinted history，甚至不必删除旧 source prefix。**

但这不是历史状态修复。tail-stitched 相对 tail-native 的 source-action leakage 中位数仍为正：
content 各格约 0.027--0.104，position 各格约 0.120--0.448。也就是说 target action 已恢复，
source alternative 的相对概率仍被抬高；late policy 是 readout override，而不是构造了
$KV_t(H)$。另一个显式写有 `overrides earlier rule` 的较长 tail 反而在 position-first 出现
1/25 tail-stitched failure 和 2/25 keep-source failure，说明 prompt wording 仍会改变竞争结果，
不能把单一强措辞当成普遍保证。

这个对照必须成为后续方法的强 baseline，也迫使论文命题进一步收窄：当前结果不能主张
“任何 agent switch 都必须修 KV”。需要关注的是无法通过末尾重申完整解决的场景，例如 target
tool schema/permission/role 不能被低优先级 user 消息重定义、长历史使 late instruction 失效，
或者要求恢复 target-native 分布而不只是当前一步 action EM。

下面的旧实验表中 failure 指 `B != A`，`B=S` 指 hybrid 完整 tool call 与 source-native 完全
相同。这些都是 pilot 样本数，尚未用于最终置信区间。

| 模型 / 数据 / policy | 全量方向 | 严格有效 | failure | `B=S` | leakage > 0 | full replay=A |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B，随机合成 history selection | 32 | 18 | 14/18 | 14/18 | 18/18 | 18/18 |
| Qwen3-14B，随机合成 history selection | 16 | 6 | 6/6 | 6/6 | 6/6 | 6/6 |
| Llama-3.1-8B，顺序码合成 pilot | 4 | 4 | 2/4 | 1/4 | 4/4 | 4/4 |
| Qwen3-8B，BFCL tagged，strong | 18 | 14 | 6/14 | 6/14 | 14/14 | 14/14 |
| Qwen3-8B，BFCL verbatim，strong | 10 | 6 | 3/6 | 2/6 | 6/6 | 6/6 |
| Qwen3-14B，BFCL tagged，strong | 16 | 16 | 0/16 | 0/16 | 14/16 | 16/16 |
| Qwen3-14B，BFCL tagged，minimal-v2 | 10 | 8 | 4/8 | 4/8 | 8/8 | 8/8 |

Qwen3-8B BFCL tagged/strong 的 14 个严格样本来自 5 个 notetaker 对话和
customer/healthcare 各一个长对话，历史为 247--1,514 tokens。finance/student 两个更长
case 中 8B 抄写 opaque ID 不稳定，未进入严格集合。Qwen3-14B strong 的 16 个严格样本覆盖
247--2,162 tokens。

Llama-3.1-8B 需要兼容其原生裸 JSON `parameters` 工具格式，以及把 tool schema 放进第一条
user message 的 chat template；runner 已用单 token marker 做 token-exact 边界定位。随机
opaque-code 扩展只有 2/16 个 case 满足 A/S 双原生正确，因此排除，不用它估计 failure rate。

### 边界一：模型规模与 target policy 强度

14B 在 strong policy 下 16/16 保持 A，但 source margin 仍在 14/16 上升；同一模型、同一
BFCL 历史、同一工具改成较短但仍可原生执行的 minimal-v2 policy 后，8 个严格方向中有
4 个直接 `B=S`。这说明大模型并非没有 imprint，而是更强的 target prefix 能在 readout 时
覆盖它。

minimal-v1 曾删除“只输出 ID、不要正文”的输出约束，导致两种模型复制整条记录并在
64-token 上限被截断。该配置测到的是 task invalidity，已明确排除，不纳入任何表格。

### 边界二：first/last 的方向先验

自然 BFCL 样本存在明显不对称。14B minimal-v2 的四个严格 reverse case 均为
`source=last, target=first, B=last`，四个 forward case 均保持 target=last；8B strong 也有类似
趋势，但存在一个 forward 反例。这意味着 source imprint 会与模型的 recency/position prior
叠加。正式实验必须双向镜像、随机化证据位置，并按方向分层报告，不能把全部错误直接解释为
source policy。

### 边界三：简单 policy-only 动作通常更稳健

早期 32-cell 受控实验中，直接指定 tool name 和固定 JSON argument 都是 0/8 behavioral
failure；history-conditioned selection 为 5/8，tool-schema 为 1/8。BFCL 短历史 specialist
function routing 也大多保持 target。当前证据支持的是“policy 改变了共享历史的解释/选择”这一
较窄命题，而不是“只要 system prompt 不同，任意 transferred KV 都会坏”。

## 因果瓶颈定位

### 1. token 区域

在 Qwen3-8B 合成 history-selection case 中，只换最后 query 的 source KV 不改变 A；只换旧
user history 可直接得到 S，assistant acknowledgement 单独无害。

在新增的两个非位置 content-third 正例中结论相同：只换 15-token final query 保持 A，只换
prior user messages 则直接得到 S。这排除了动作翻转仅由最后指令 token 或 first/last 位置操作
触发的解释。

在 BFCL 自然 case 中，最后 18-token batch trigger 在 8B/14B 上也都无害；整个 prior history
可直接得到 S。但 user-only 或 assistant-only 只提高 source margin，不足以单独翻转。这表明
自然对话中的错误是跨消息组合状态，而不是最后 query 的局部污染。

### 2. K/V 组件

- 合成 history selection：8B 和 14B 的全层 V-only 多次足以得到 S，K-only 基本保持 A；
- 自然 BFCL：K-only 和 V-only 都只造成连续概率漂移，组合后才离散翻转。
- 非位置 content-third 的两个 8B 正例：全层 K-only 均直接得到 S，V-only 均保持 A。

所以“V 是主要载体”已经被新的反例明确否定为普遍结论。不同 policy family 可以主要改变
attention routing（K）、被读取内容（V），或两者的协同；方法不能预先固定只修 K 或只修 V。

### 3. 层区间

| 模型 / case | 能控制行为的连续区间 | 相对深度 | 因果结果 |
|---|---:|---:|---|
| Qwen3-8B，合成 | 18--23 / 36 | 50%--67% | source-only 得 S；修回得 A |
| Qwen3-8B，BFCL natural | 18--23 / 36 | 50%--67% | source-only 得 S；修回得 A |
| Qwen3-8B，content-third，2 cases | 18--29 / 36 | 50%--83% | source-only 得 S；联合修回得 A |
| Qwen3-14B，合成 | 25--29 / 40 | 62.5%--75% | source-only 得 S；修回得 A |
| Qwen3-14B，BFCL natural | 20--29 / 40 | 50%--75% | source-only 得 S；修回得 A |
| Llama-3.1-8B，合成 | 12--15 / 32 | 37.5%--50% | source-only 得 S；修回得 A |

14B natural 的 5-layer patch 表明修回 20--24 或 25--29 任一块都足以恢复 A，但单独放入
任一 5-layer source block 不足以翻转；10-layer patch 则证明 20--29 联合既充分又必要。
新的 content-third 重复给出互补结构：两个 case 的 source-only 18--23 都足以翻转，其中一个
source-only 24--29 也足以翻转；但单独 repair 任一 6-layer 子块可能仍为 S，联合 repair
18--29 在两个方向都恢复 A。说明中层 policy state 可能有冗余承载，修复目标应覆盖联合区间，
不能只根据单块 sufficiency 选择一个 6-layer block。
这使当前最稳定的 mechanistic bottleneck 成为：**模型中层形成的策略条件化历史状态**。
Qwen 内部落在约 50%--75% 深度，但 Llama 正例提前到 37.5%--50%；具体层位是架构相关的，
方法必须学习 layer gate，不能硬编码 Qwen 的层比例。

### 4. replay correction budget

Qwen3-8B 的 BFCL natural reverse failure 共 389 history tokens：

| target-native suffix replay | 0 | 16 | 32 | 64 | 128 | 192 | 256 | full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 行为 | S | S | S | S | S | A | A | A |

192-token suffix 从第三条 user record 中间开始，而最后 trigger 只有 18 tokens。它说明只重算
handoff trigger 或最后一轮不能修复；但这个 case 也不需要全历史 prefill。真正需要估计的是
每次 handoff 的最小 counterfactual correction span，而不是固定 replay N tokens。

## 当前确认的三个问题

1. **语义问题**：source 历史 KV 不是 target policy 下应有的反事实状态；re-RoPE 只能修位置，
   不能消除 source policy 对历史解释的影响。
2. **评测问题**：长历史 first/last retrieval 的 A/S task validity 会随模型、prompt 和输出长度
   变化。没有 eligible gating，会把基础能力错误误报成 KV 错误。
3. **方法问题**：错误不是一个全层统一线性坐标变换。关键状态集中在中层，但自然任务又表现出
   K/V、user/assistant 的组合效应，单独修 V 或最后几个 tokens 不够稳健。

## 对方法设计的直接启示

下一版方法不应直接学习“source KV → target KV”的全缓存 mapper，而应优先验证：

- 先用少量 probe 自动选择每个模型的关键层，再只在这些层预测 target-policy correction
  residual；
- correction 同时条件化 target policy/tool schema 的表示，而不是只有 source KV；
- 联合修 K/V，允许 learned gate 判断某层/某 token 是否需要 correction；
- 将 selective replay 作为 oracle：先用最小 suffix 或消息级 replay 找到需要修复的 span，再训练
  learned correction 逼近该 counterfactual state；
- 用 source leakage probe 作为运行时风险分数：强 target policy 且风险低时直接复用，风险高时
  correction 或 replay。

在论文层面，当前最可辩护的主张是“agent handoff 需要 policy-conditioned counterfactual KV
correction”，不是“已有 KV reuse 在所有 agent 场景都失效”。下一阶段需要把自然严格样本扩到
至少数百个，并加入 tool-schema/role/permission 冲突，才能估计真实发生率和方法收益。

## 主要产物

- runner：`xkv-policy-imprinting`、`xkv-policy-triad`、`xkv-bfcl-memory-policy-imprinting`；
- triad 汇总器：`xkv-summarize-policy-triad`；
- triad n=25 原始分片：
  `outputs/policy_imprinting/qwen3_8b_policy_triad_n25_seed2027_shard{0,1}.jsonl`；
- triad n=25 结构化汇总：
  `outputs/policy_imprinting/qwen3_8b_policy_triad_n25_seed2027_summary.json`；
- triad content-third patch（两个相反方向）：
  `outputs/policy_imprinting/patching_qwen3_8b_triad_content_{urgent,cheapest}_*block18_29.json`；
- Qwen3-14B triad pilot 汇总：
  `outputs/policy_imprinting/qwen3_14b_policy_triad_pilot_seed41_summary.json`；
- plain late-policy n=25 汇总：
  `outputs/policy_imprinting/qwen3_8b_policy_tail_plain_n25_seed2027_summary.json`；
- explicit-override late-policy n=25 汇总：
  `outputs/policy_imprinting/qwen3_8b_policy_tail_n25_seed2027_summary.json`；
- causal localization：`xkv-policy-patching`；
- 8B natural patch：
  `outputs/policy_imprinting/patching_qwen3_8b_bfcl_memory_notetaker0_tagged_reverse.json`；
- 14B natural patch：
  `outputs/policy_imprinting/patching_qwen3_14b_bfcl_memory_notetaker0_tagged_minimal_reverse_block10.json`；
- Llama-3.1-8B patch：
  `outputs/policy_imprinting/patching_llama31_8b_history_forward_h32.json`；
- replay curve：
  `outputs/policy_imprinting/qwen3_8b_bfcl_memory_notetaker0_tagged_reverse_replay_curve.jsonl`。
