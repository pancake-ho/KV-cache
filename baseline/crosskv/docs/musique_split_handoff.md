# MuSiQue Split-Hop A→B handoff benchmark

## 数据构造

输入使用根目录 `musique/` 下的官方 MuSiQue-Answerable v1.0。评测主体只来自 dev；train
只提供同关系的干扰记录，不拟合任何模型或 KV 参数。

每条 dev 两跳问题被拆成：

1. Agent A 阅读第一跳支持段落和真实 MuSiQue 干扰段落，回答中间桥接实体；
2. Agent B 接收八条具有相同关系的候选记录，根据 A 的桥接实体返回另一条最终答案。

八条候选具有八个不同 `LOOKUP_KEY` 和八个不同 `RESULT`，所以未得到 A 结果时结构化随机
基线为 1/8。候选的实体、答案和 evidence 均来自 MuSiQue，而八选一布局属于受控的半合成
handoff；不得把它报告为原始 MuSiQue 准确率。

严格筛选后得到 197 条而不是硬凑 200 条。每条 A dossier 至少 24,000 字符：

| tokenizer | 最短 | 中位数 | 最长 | >=4K |
|---|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 5,274 | 5,703 | 6,500 | 197/197 |
| Qwen3-8B | 5,509 | 5,966 | 7,334 | 197/197 |

冻结数据：`datasets/musique_handoff/dev197_seed2027.jsonl`。

## 实验臂

- `B-only`：不给 A 中间答案，只给八条候选；验证 A→B 依赖。
- `Oracle-summary`：显式给 gold bridge；测 B lookup 上界。
- `Summary-only`：只给 A 的实际文本输出；这是判断完整历史 KV 是否必要的关键强基线。
- `Native-front`：B system prompt 在最前，B 对完整 A dossier、A 输出和 B 候选做 dense prefill。
- `Native-tail`：保留 A 的原始 system/history，在末尾附加 B policy 和新任务并 dense prefill。
- `Tail-KV`：与 Native-tail 完全相同的 token 序列，但直接从 A 的 live KV 续接，只 prefill B
  suffix。它是 exact continuation，不含 KVComm offset/anchor，不能称为 KVComm。

## 20 条协议 pilot

模型为 Llama-3.1-8B-Instruct，物理 GPU2，贪心生成。这里只用于检查协议，不作为正式论文数字。

| arm | 最终正确率 | B 端平均 prefill tokens |
|---|---:|---:|
| B-only | 0/20 | 1,170.00 |
| Oracle-summary | 19/20 | 1,168.85 |
| Summary-only | 18/20 | 1,170.15 |
| Native-front | 14/20 | 7,151.85 |
| Native-tail | 16/20 | 7,229.85 |
| Tail-KV | 16/20 | 1,167.00 |

A 的严格短答案 EM 为 13/20，但 19/20 输出包含正确桥接实体；差异主要是
`Corfe Mullen, Dorset, England` 一类扩写。B-only 0/20 与 Oracle 19/20 表明 A→B dependency
已经建立。Native-tail 与 Tail-KV 在 20/20 上逐字一致，验证 exact cache continuation 的实现；
Tail-KV 相对同序列 dense prefill 少约 83.9% 的 B 端 prefill tokens。

目前最重要的结果不是 14/20 与 16/20 的小样本高低，而是 Summary-only 达到 18/20：对于只需
一个桥接实体的任务，传递短文本比携带整个 5K+ A 历史更便宜且更准。因此该 track 可以用来测
handoff dependency 和 source-policy 干扰，但不能单独证明“必须复用完整长历史 KV”。正式工作
必须保留 Summary-only，并另设一个 B 需要核验 A 原始证据、短摘要不是充分统计量的 track。

## 复现文件

- builder：`src/xmodel_kv/musique_handoff.py`
- builder CLI：`src/xmodel_kv/cli/prepare_musique_handoff.py`
- Tail runner：`../KVCOMM/experiments/run_musique_handoff_tail.py`
- pilot：`../KVCOMM/runs/musique_handoff/tail_pilot_n20_structured/`

CrossKV 单元测试：85/85 通过。
