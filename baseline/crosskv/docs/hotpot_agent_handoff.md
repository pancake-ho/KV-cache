# HotpotQA 双智能体 handoff：A 历史 KV 复用预实验

本文档记录 2026-08-10 完成的真实任务级预实验。它不再把一段外部构造的固定历史放在两个
互斥 system prompt 下，而是让 Agent A 先实际读取任务并生成报告，再让 Agent B 基于 A 的
完整输入/输出给最终答案。

## 应用流程

模型固定为本地 Qwen3-8B，数据使用 LongBench 的 200 条 HotpotQA：
`/data/datasets/longbench/hotpotqa.jsonl`。每题的 passages 和 question 构成共享任务。

1. Agent A 是 evidence analyst：读取所有 passages 和 question，找出两跳证据链、桥接实体和
   相关事实，生成不超过 256 token 的研究报告及候选答案。
2. Agent B 是 final answerer：接收原始 passages/question、A 的报告，以及最后一条
   `Return the final answer ... Output only the short answer.` 指令，只输出短答案。
3. 因而 B 接收的共享历史确实是 `A 输入 + A 输出`，最后的 B 指令在共享历史之后原生计算。

每题包含四个实验臂：

- `Direct-B`：B 独立读取原始任务，不接收 A 报告；用于测量协作收益。
- `Native-AB`：B 在自己的 system prompt 下从头 prefill 原始任务、A 报告和最终指令；准确率上界。
- `Identity-AB`：仍使用 B 自己形成的共享历史 KV，但按与复用完全相同的切片、拼接和分块路径
  执行；用于估计纯执行/数值误差。
- `Reuse-AB`：把在 A system prompt 下形成的 `原始任务 + A 报告` 历史 KV 切出、re-RoPE 后
  接到 B 的原生 prefix；最后 B 指令原生计算，不重算共享历史。

`Reuse vs Identity` 是 policy imprinting 的主因果对照，因为两者的执行路径、token、分块和
拼接完全相同，唯一差别是共享历史 KV 在 A 还是 B 的角色下形成。`Reuse vs Native` 是实际
部署端到端差距，但其中还包含长序列分块计算的浮点漂移。

## 200 题结果

所有系统使用贪心解码。HotpotQA 指标采用归一化 exact match 和 token F1；差值区间来自
20,000 次 paired bootstrap。

| 实验臂 | EM | F1 |
|---|---:|---:|
| Direct-B | 42.50 | 56.99 |
| Native-AB | 50.50 | 65.74 |
| Identity-AB | 50.00 | 65.58 |
| Reuse-AB | 49.50 | 65.14 |

成对差值：

| 对比 | EM 差值（95% CI） | F1 差值（95% CI） |
|---|---:|---:|
| Native-AB − Direct-B | +8.00 `[+2.50,+13.50]` | +8.75 `[+3.57,+14.02]` |
| Reuse-AB − Native-AB | -1.00 `[-3.00,+1.00]` | -0.60 `[-1.74,+0.43]` |
| Reuse-AB − Identity-AB（主对照） | -0.50 `[-2.50,+1.50]` | -0.43 `[-1.60,+0.66]` |
| Identity-AB − Native-AB | -0.50 `[-1.50,0.00]` | -0.17 `[-0.50,0.00]` |

Native-AB 相对 Direct-B 的增益显著，说明 A 的报告确实使这个双智能体流程变好，不是为了
测试 KV 而加入的无用角色。Reuse-AB 与 Identity-AB 的答案一致率为 93.0%（Wilson 95% CI
88.6%--95.8%），即 A-role KV 使 14/200 个最终答案文本发生变化；但净准确率下降很小且置信
区间跨零。EM 翻转为 3 个 `Identity correct → Reuse wrong` 和 2 个反向，exact McNemar
`p=1.0`。因此，这个 200 题实验能证明实际 handoff 中存在可观察的 source-role 行为偏移，
但不能证明它在普通 HotpotQA 上造成显著的平均准确率损失。

Identity 与一次性 Native 在 199/200 题答案一致。唯一差异是第 144 题的 `35` 与
`35 people`；两者都来自 B-policy KV，说明这是 17K token 附近分块 prefill 的数值边界效应，
不能归因于 policy imprinting。这也是主比较必须使用 Identity 而非只用 Native 的原因。

按共享历史长度分层，`Reuse − Identity` F1 分别为：`<8K` +1.07 pp（n=25）、`8--12K`
-0.94 pp（n=32）、`12--16K` -0.06 pp（n=60）、`>=16K` -0.96 pp（n=83）。当前没有单调的
长度退化结论。

## 行为样例

三个 source imprint 导致 EM 下降的题都表现为 Reuse 更倾向复制 A 报告的完整候选表述：

- index 6，gold/Identity 为 `Tharangambadi`，Reuse 输出
  `Thirukkalacherry is a village in the Indian town of Tharangambadi.`；
- index 75，gold/Identity 为 `14`，A 报告候选是 `14 points`，Reuse 也输出 `14 points`；
- index 29，gold/Identity 为 `University of Southern California`，Reuse 复制 A 的
  `University of Southern California (USC)`。

反向改善也具有同样机制。index 22 中 A 报告候选为正确的 `Long Island`，Identity 输出错误的
`Jones Beach Island`，Reuse 则复制 A 候选而变为正确。也就是说，当前观测不是无方向的
cache 损坏，而更像 B 对 A 角色中形成的报告/答案表述产生了额外依赖；当 A 对时它可以帮助，
当 A 的格式或结论不符合 B 的目标时才会伤害。

## 当前结论和下一步

这个 benchmark 已满足真实应用叙事：A 实际完成一个子任务并输出，B 接收 A 的全部任务历史，
且 Reuse 避免重算中位数 15,152 个共享 token；按 token 数估计，平均复用了 B prefill 的
99.11%。但这只是语义准确率实验：当前 source cache 是从已冻结的 A transcript 重新 prefill
重建的，并非在 A 在线生成时直接截获，因此还没有测量真实 latency 或显存收益。

HotpotQA 的当前 A/B 目标高度一致：A 已经给出最终候选，B 主要做短答案格式化，所以 source
imprint 既可能伤害也可能帮助，平均效应互相抵消。它适合作为第一个自然任务 pilot 和论文中的
“aligned handoff”分层，不足以单独支撑“复用会显著降低最终准确率”的主表结论。

下一版应继续使用 HotpotQA，但把协作职责变成真实的非对称分工：A 只定位第一跳桥接实体和
支持句，明确不解答最终问题；B 使用桥接结果完成第二跳并输出最终答案。这样 A 的工作仍然有用，
但 A 的 readout policy 与 B 的最终任务不再同构。随后再加入 verifier handoff（A 提候选，B 必须
检查并在必要时纠错）和多模型/多次冻结报告复现，才能判断 source policy 是否造成稳定且显著的
任务准确率损失。

## 复现文件

- runner：`src/xmodel_kv/cli/hotpot_agents.py`
- metric：`src/xmodel_kv/hotpotqa.py`
- summarizer：`src/xmodel_kv/cli/summarize_hotpot_agents.py`
- 原始分片：`outputs/hotpot_agents/qwen3_8b_hotpot_agents_full200_shard{0,1,2}.jsonl`
- 正式汇总：`outputs/hotpot_agents/qwen3_8b_hotpot_agents_full200_summary.json`

运行命令可通过 `uv run xkv-hotpot-agents --help` 和
`uv run xkv-summarize-hotpot-agents --help` 查看。当前测试为 52/52 通过。
