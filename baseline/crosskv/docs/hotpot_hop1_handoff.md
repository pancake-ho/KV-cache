# HotpotQA 非对称 hop1→final KV handoff 实验

本文档记录 2026-08-11 完成的第二版自然任务实验。相比
[`hotpot_agent_handoff.md`](hotpot_agent_handoff.md) 中 A、B 都面向最终答案的 aligned handoff，
本协议让两个 agent 承担不同职责，主要指标仍是 Agent B 的最终 HotpotQA EM/F1。

## 协作协议

- Agent A 是受限的 first-hop specialist，只能输出
  `BRIDGE / EVIDENCE / NEXT_LOOKUP` handoff memo，负责桥接实体和第一跳支持事实；system prompt
  明确禁止它提交团队最终答案。
- Agent B 是 final second-hop solver，接收原始 passages/question 和 A memo，独立核查并完成
  剩余证据链，最后只输出短答案。
- B 的最终 query 位于 A 输入/输出之后并原生计算。因此该协议已经包含“把 target 指令放在最后”
  这个强 baseline；测试的是 late instruction 之后仍然存在的 KV 条件差异。

每题运行五个 arm：

- `Direct-B`：B 不看 A memo；
- `Source-AB`：共享历史和最后 query 都在 A policy 下读取；
- `Native-AB`：B 在自己的 policy 下从头 prefill 全部历史；
- `Identity-AB`：B-policy history KV 走与 Reuse 相同的分块和 stitch 路径；
- `Reuse-AB`：A-policy 下形成的 `任务 + A memo` 历史 KV re-RoPE 后接到 B-native prefix，最后
  B query 原生计算。

主因果对照是 `Reuse-AB − Identity-AB`。`Native-AB − Direct-B` 用于确认双 agent 协作本身
确实有用；`Identity-AB − Native-AB` 单独量化分块 prefill 的数值误差。

## 数据完整性

使用本地 LongBench `hotpotqa.jsonl` 200 行和 `hotpotqa_e.jsonl` 300 行。两者 `_id` 不重叠，
但按 case-fold、空白归一化后的 question 检查发现 10 个问题文本重复。因此：

- standard-200 和 E-300 分别报告；
- pooled 结果先保留 standard 行，再删除 E 中 10 个重复 question，独立分析集为 490 行；
- 不使用未经去重的 500 行结果作为论文数字。

A memo 在 500 个原始执行行中 100% 符合指定 schema。去重后的 490 题中，55.31% 的 memo 文本
直接包含 gold；这不是数据泄漏，而是某些 HotpotQA 桥接实体或第一跳事实本身就包含答案字符串，
因此下面另外给出条件结果。

## 最终准确率

### 分 split 结果

| split / arm | Direct-B | Native-AB | Identity-AB | Reuse-AB |
|---|---:|---:|---:|---:|
| standard-200 EM | 41.50 | 48.00 | 48.00 | 46.00 |
| standard-200 F1 | 55.18 | 61.07 | 60.95 | 60.48 |
| E-300 EM | 46.67 | 53.67 | 53.33 | 52.33 |
| E-300 F1 | 60.60 | 67.78 | 67.44 | 66.64 |

两个 split 的协作收益方向一致且显著：

- standard：Native−Direct EM +6.50 pp，95% CI `[+1.00,+12.00]`；F1 +5.88 pp，
  `[+0.34,+11.45]`；
- E：Native−Direct EM +7.00 pp，`[+2.67,+11.67]`；F1 +7.18 pp，
  `[+3.04,+11.30]`。

Reuse 相对 Identity 都是负方向，但单个 split 均未显著：

- standard：EM −2.00 pp，`[-4.50,0.00]`；F1 −0.46 pp，`[-2.93,+1.93]`；
  EM 翻转 5 个 correct→wrong、1 个反向，exact McNemar `p=0.21875`；
- E：EM −1.00 pp，`[-3.33,+1.33]`；F1 −0.81 pp，`[-2.70,+1.04]`；
  翻转 8:5，`p=0.58105`。

### 去重 pooled-490

| arm | EM | F1 |
|---|---:|---:|
| Direct-B | 44.29 | 58.17 |
| Native-AB | 51.22 | 65.03 |
| Identity-AB | 51.02 | 64.78 |
| Reuse-AB | 49.80 | 64.21 |

- Native−Direct：EM +6.94 pp，95% CI `[+3.47,+10.41]`；F1 +6.86 pp，
  `[+3.56,+10.23]`；
- Reuse−Identity：EM −1.22 pp，`[-2.86,+0.41]`；F1 −0.57 pp，
  `[-2.05,+0.87]`；
- Identity correct→Reuse wrong 为 12，反向为 6，exact McNemar `p=0.23788`；
- Reuse 与 Identity 的最终答案一致率为 87.76%，即 60/490 个答案发生改变；
- Identity 与一次性 Native 一致率为 99.18%，且 EM 只差一个样本，说明主效应不是普通 stitch
  实现退化。

所以当前严格结论是：**非对称 handoff 把 KV-sensitive answer change 从 aligned-200 的 7.0%
提高到约 12.2%，两个独立 split 的平均准确率差值也都为负；但 490 题上的平均净损失仍未达到
统计显著，不能写成“朴素复用显著降低总体 HotpotQA 准确率”。**

## 长历史探索性结果

长度格 `<8K / 8--12K / 12--16K / >=16K` 在运行本实验前已存在于 summarizer，但这里仍涉及
四个分层比较，必须作为探索性结果而非确认性主结论。

去重 pooled-490 的 `>=16K` 格包含 125 题：

- Identity EM 42.40，Reuse EM 38.40，差值 −4.00 pp，paired bootstrap 95% CI
  `[-8.00,-0.80]`；
- EM 翻转为 5 个 correct→wrong、0 个反向，exact McNemar `p=0.0625`；
- Identity F1 59.10，Reuse F1 56.73，差值 −2.38 pp，95% CI `[-6.09,+0.90]`。

去重前错误地把一个跨 split 重复题算了两次，会得到 6:0 和 `p=0.03125`；该数字已经废弃，
不得引用。去重后的 exact test 尚未越过 0.05，且四格多重比较会进一步削弱证据。它目前只支持
“长历史可能放大平均损失”的后续假设，需要新的独立样本确认。

五个独立的长历史 EM 下降中既有格式问题，也有真正的证据选择错误：

- `Tharangambadi` 被扩写为整句；
- 完整原因被缩成 `Alzheimer's disease`；
- 正确地点 `West Lafayette, Indiana` 变成 A memo 中的错误桥接对象 `Houston Oilers`；
- 正确语言 `Berber` 变成相邻实体 `Siwi`；
- 正确皇帝 `Caligula` 变成 lineage 中的另一个实体 `Claudius`。

后三个例子不能用 short-answer 格式偏差解释，说明复用确实会改变 B 对长上下文证据/实体的选择。

## Source 跟随与条件分析

去重后的 490 题里，Source readout 与 Identity 答案不同的有 216 题。在这些题中，Reuse 精确
匹配 Identity 162 次、匹配 Source 11 次、两者都不匹配 43 次。因此自然任务中的 effect 不是
简单的 `B=S`：A-policy KV 改变了 B 的条件分布，通常 target late query 仍占主导，少数题直接
追随 source，另一些题转到第三个答案。

按 A memo 是否直接包含 gold 分层：

| A memo | n | Identity EM | Reuse EM | EM 差 | Identity F1 | Reuse F1 | F1 差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| contains gold | 271 | 67.53 | 66.05 | −1.48 | 78.57 | 77.37 | −1.19 |
| no gold string | 219 | 30.59 | 29.68 | −0.91 | 47.72 | 47.93 | +0.21 |

直接含 gold 的 memo 中准确率下降更一致，但该分层由生成内容事后定义，不能当作随机化因果 arm。

## 系统边界和下一步

共享历史中位数为 11,879 tokens；按 token 数，Reuse 平均覆盖 B prefill 的 98.45%、中位数
98.91%。这些数字只是潜在计算量覆盖，不是 latency 测量。当前 A cache 仍从冻结后的 A transcript
重建，用于稳定的语义准确率对照；尚未在 A 在线生成时直接截获 live KV。

这个 benchmark 现在适合承担三种论文角色：

1. 证明自然双智能体协作确实提高任务准确率；
2. 测量 late target instruction 后仍存在的 KV-sensitive answer change；
3. 提供长历史和具体证据选择失败的自然案例。

它目前还不够单独证明显著的总体准确率损失。下一步最有效的不是继续调 prompt，而是下载原始
HotpotQA validation 集，冻结至少 1,000--2,000 个 A memo 后做预注册的 `>=16K` confirmatory
test；同时实现 live KV capture。若要测试更强且自然的 policy 冲突，可增加 verifier handoff：
A 提候选，B 的职责是识别并纠正错误候选，主指标按 A 正确/错误分层报告。

## 复现文件

- runner：`src/xmodel_kv/cli/hotpot_agents.py --protocol hop1`
- summarizer：`src/xmodel_kv/cli/summarize_hotpot_agents.py`
- standard 原始输出：`outputs/hotpot_agents/qwen3_8b_hotpot_hop1_strict_full200_shard{0,1,2}.jsonl`
- E 原始输出：`outputs/hotpot_agents/qwen3_8b_hotpot_hop1_strict_e300_shard{0,1,2}.jsonl`
- standard summary：`outputs/hotpot_agents/qwen3_8b_hotpot_hop1_strict_full200_summary.json`
- E summary：`outputs/hotpot_agents/qwen3_8b_hotpot_hop1_strict_e300_summary.json`
- 去重 pooled summary：
  `outputs/hotpot_agents/qwen3_8b_hotpot_hop1_strict_pooled490_dedup_summary.json`

当前单元测试 55/55 通过。
