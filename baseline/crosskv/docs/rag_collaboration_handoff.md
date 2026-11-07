# RAG 与多智能体协作 KV 复用预实验

本文档记录 2026-08-14 新增的两个 agent-switch 场景。它们的目标不是直接作为论文主表，
而是回答两个问题：

1. `P_B + KV_A(H)` 在长 RAG/协作历史中是否仍会精确追随 A；
2. 把完整 `P_B` 放到末尾，或者把 `P_B + 当前请求` 一起 late-bind，能否低成本解决问题。

## 场景 1：policy-conditioned RAG

数据来自本地 LongBench HotpotQA。每例保留真实问题和检索 passages，但把原答案实体在两个
检索 namespace 中分别匿名化为两个不同的 opaque `EVID-` ID。A/B system policy 分别授权一个
namespace；共享历史 H 包含两个版本，token 序列完全相同。匿名化用于阻止模型依靠参数知识
直接回答，所以这是 LongBench-derived 的半合成 RAG benchmark，不能称为原始 HotpotQA 准确率。

10 例 pilot 的共享历史为 6,831--16,565 tokens，均值 10,792.9。

## 场景 2：planner → isolated executor

每例使用 8 条真实 BFCL `simple_python` 请求正文，合成 tenant、execution lane、priority 和
长事件日志。Agent A 读取约 4.8K tokens 的事件/背景历史，并输出中性的规范化 current-job
table；Agent B 接管后，根据自己的 tenant 分别选择 RED/BLUE lane 中最高优先级 job，并调用
`dispatch_job_batch`。组合动作空间为 `C(8,2)=28`，工具参数按无序二元集合判分。

请求正文来自 BFCL，但状态和 policy 是合成的，因此应称为 BFCL-derived collaboration handoff。

## 实验臂

- `source_front_native`：`P_A + plaintext(H)`；
- `front_full_reprefill`：`P_B + plaintext(H)`，目标上界；
- `front_identity`：B 自己的 H-KV 经过同一拼接路径；
- `front_kv_reuse`：`KV_B(P_B) + KV_A(H)`；
- `tail_full_reprefill`：`neutral + plaintext(H) + P_B`；
- `tail_kv_reuse`：`KV_neutral(prefix) + KV_A(H) + native(P_B)`；
- `tail_request_*`：在上一组基础上同时 late-bind B 的当前短请求，是最强协议重设计基线。

严格集合要求 Source Native、Target Native 和 Front Identity 都正确。Tail 组另做 layout-specific
配对：只有对应 Tail plaintext 正确时，才能把 Tail KV 的错误解释为 KV-specific failure。

## 10+10 pilot 结果

### Qwen3-8B

| 场景 | 全体 Target Native | 严格 n | Front reuse 执行 B | Front reuse 追随 A | Tail+request plaintext → reuse |
|---|---:|---:|---:|---:|---:|
| RAG | 6/10 | 4 | 0/4 | 4/4 | 4/4 → 3/4 |
| Collaboration | 5/10 | 3 | 0/3 | 3/3 | 3/3 → 2/3 |

RAG 的一例 Tail+request KV 失败精确输出 A 的 `EVID`；协作的一例输出一个 A job 和一个 B job，
不是完整追随 A。

### Qwen3-14B

| 场景 | 全体 Target Native | 严格 n | Front reuse 执行 B | Front reuse 追随 A | Tail+request plaintext → reuse |
|---|---:|---:|---:|---:|---:|
| RAG | 4/10 | 4 | 0/4 | 4/4 | 4/4 → 3/4 |
| Collaboration | 8/10 | 7 | 0/7 | 7/7 | 6/7 → 6/7 |

14B RAG 的唯一 Tail+request 失败生成了目标 ID 加多余后缀，不是追随 A。协作中，完整 policy
放末尾时 7 个 layout-correct 严格样本有 6 个恢复；追加当前请求没有产生额外 KV-specific 错误。

## 当前结论

两个新场景都稳定复现了前置 KV 复用的 policy imprinting：合并两个模型的严格集合，
`front_kv_reuse` 为 0/18 执行 B、18/18 精确追随 A。Identity 全部保持 Target Native，排除了
拼接和 re-RoPE 本身导致的一般退化。

但更重要的负面结论是：strong late-binding 仍然解决了大多数样本。对 Tail+request 自身可解的
17 个严格/layout-correct 样本，KV reuse 恢复 14/17；只有 1/17 精确追随 A，另外两例分别为
混合 job 和格式错误。因此，这批结果支持“前置标准协议下存在严重污染”，但仍不足以支持
“完整 B policy 放末尾也普遍无效”。多智能体的规范化表越靠近 handoff，tail 修复越容易；RAG
长证据中的 residual gap 更值得继续扩大。

## 复现

只使用物理 GPU2：

```bash
CUDA_VISIBLE_DEVICES=2 python -m xmodel_kv.cli.evaluate_handoff_scenarios \
  --scenario both \
  --model Qwen/Qwen3-8B \
  --output outputs/handoff_scenarios/qwen3_8b_pilot_rag10_collab10_v2.jsonl \
  --rag-cases 10 --collaboration-cases 10 --background-records 36 \
  --tail-role system --device cuda:0 --max-new-tokens 128
```

代码：`src/xmodel_kv/handoff_scenarios.py`、
`src/xmodel_kv/cli/evaluate_handoff_scenarios.py`；逐样本和 summary 位于
`outputs/handoff_scenarios/`。
