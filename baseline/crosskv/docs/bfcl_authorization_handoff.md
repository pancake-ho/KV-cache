# Agent KV 复用的跨租户授权交接实验

本文档记录 2026-08-11 完成的应用化预实验。目标是寻找一个同时满足以下条件的 agent handoff：

1. Agent A 确实读取任务并产生有用但未执行的输出；
2. Agent B 接收 A 的输入和输出，但必须执行自己的 policy；
3. A/B policy 存在真实系统中的硬冲突，错误不能被“平均准确率互相抵消”掩盖；
4. 输出可以自动、确定性判分；
5. 复用失败必须定向追随 A，而不是一般性的 cache surgery 退化。

## 为什么放弃两个较自然但证据弱的协议

HotpotQA evidence-analyst → final-answerer 的 200 题实验中，Native-AB/Reuse-AB 的 EM 分别为
50.5%/49.5%，主因果对照 Reuse-AB − Identity-AB 只有 -0.5 pp。A/B 的目标高度一致，source
imprint 有时伤害、有时帮助，净效应抵消。进一步把 B 改成 verifier 也没有形成稳定的行为冲突：
同一个模型被重新命名为 verifier 并不自动带来新的证据或更强的纠错能力。

BFCL planner → executor 的第一版曾出现显著下降，但最后的 handoff trigger 当时也错误地随 A
的 cache 一起转移。把完整 trigger 放回 B 侧原生计算后，Qwen3-14B 从 24/67 vs 16/67
恢复到 24/67 vs 24/67；因此早期显著性不能作为论文证据。无 trigger 的自动切换版本同样没有
净退化。这些负结果说明：普通 aligned collaboration 适合做外部有效性分层，不适合作为现象主表。

## 新场景：cross-tenant planner → isolated executor

新实验把 handoff 放在多租户 agent 服务中的硬授权边界上。每个 case 从 BFCL v4
`simple_python` 抽取 8 条真实用户请求，为每条请求加入：

- 一个随机 opaque `JOB-` ID；
- `TENANT-ALPHA` 或 `TENANT-BETA`，各 4 条；
- 一个互不相同的随机优先级。

Agent A 是一个租户的 planner/executor，读取完整队列并提出该租户最高优先级 job，但明确标记为
“not yet executed”。随后切换到另一个租户的 isolated executor B。B 必须把 A 的 proposal 当作
不可信输入，只能 dispatch 自己租户中优先级最高的 job。动作空间固定为 8，随机命中率 12.5%。

50 个独立队列同时运行 ALPHA→BETA 和 BETA→ALPHA，共 100 个方向样本。方向镜像、job 位置和
优先级随机化，避免固定租户、固定位置和 first/last policy 伪影。最终动作统一为：

```text
dispatch_queued_job(job_id=<opaque ID>)
```

这是一项半合成 benchmark：8 条任务正文来自 BFCL 官方数据，但 tenant、priority、queue 和 ACL
是本实验构造的。因此结果必须称为 “BFCL-derived authorization handoff”，不能称作 BFCL 官方
准确率。真实请求正文在当前 selection rule 中不决定 gold job；它提供任务负载和现实输入分布，
ACL/priority 才决定授权动作。

## 因果实验臂

- `Source-native`：在 A 的 system policy 下完整 prefill，应该选择 A 租户的 gold job；
- `Target-native`：在 B 的 system policy 下完整 prefill 同一队列和同一 A proposal，是准确率上界；
- `Identity`：把 B 自己形成的历史 KV 用相同的切片、拼接路径接回 B prefix；
- `Reuse`：把 A 形成的历史 KV 接到 B-native prefix，不重算共享历史；
- `Post-trigger reuse`：在 A proposal 之后追加一条中性的 B-turn 消息，并把该消息的完整 26 tokens
  在 B policy 下重算；其余历史继续复用 A KV。

Target-native 与 Reuse 看到的 A proposal 文本完全相同；两者的关键差别只是共享 token 的 KV 是在
A policy 还是 B policy 下形成。Identity 保持 B 的行为，排除了切片、拼接或位置变换本身导致的
表示退化。

## 主结果

| 模型 / 历史 | 样本 | Target-native gold | Reuse gold | Native 对→Reuse 错 | 严格集合 `Reuse=Source` |
|---|---:|---:|---:|---:|---:|
| Qwen3-14B，约 652 tokens，无 trigger | 100 | 91% | 0% | 91:0，`p=8.08e-28` | 89/89 |
| Qwen3-14B，约 674 tokens，26-token B trigger | 100 | 92% | 0% | 92:0，`p=4.04e-28` | 91/91 |
| Qwen3-8B，约 652 tokens，无 trigger | 100 | 88% | 1% | 87:0，`p=1.29e-26` | 82/82 |
| Qwen3-8B，约 674 tokens，26-token B trigger | 100 | 91% | 0% | 91:0，`p=8.08e-28` | 85/85 |
| Qwen3-14B，约 8,183 tokens，无 trigger | 100 | 69% | 0% | 69:0，`p=3.39e-21` | 69/69 |
| Llama-3.1-8B，约 652 tokens，无 trigger | 100 | 24% | 0% | 24:0，`p=1.19e-7` | 17/17 |

`p` 是双侧 exact McNemar。严格集合要求 A、B 原生都命中各自预定义 gold，A/B 输出不同，且
Identity 与 Target-native 一致。Qwen 的主结果在两个方向上都成立；在严格集合中，Reuse 不是
随机选错，也不是退回 first-job 先验，而是 100% 精确输出 Source-native 的 opaque job ID。
Llama 的 B 原生准确率过低，只作为方向一致性证据，不进入 headline comparison。

Post-trigger 实验中 100/100 Identity 与 Target-native 一致；每例最后一条 26-token user message
都被完整重算。Qwen3-14B 仍在 92 个 target 可解样本中 0/92 跟随 target、92/92 跟随 source。
因此“切换后增加一句 B 的提示”不能修复这组授权交接。

长历史由 60 条 BFCL memory conversation 的已完成背景记录、8 条 active queue job 和 A proposal
组成，中位历史长度 8,182.5 tokens。零 replay 时可复用部分占 B prefill token 的中位数 96.8%。
长历史的 Target-native 降到 69%，说明当前 prompt 对无关背景还不够鲁棒；主因果结论只在 69 个
B 原生可解且 identity-valid 的样本上解释，不能把其余 31 个基础任务失败算作 KV 失败。

## Tail replay 下界

为了排除“只重算最后 proposal 就够了”，Qwen3-14B 的 100 个短历史方向样本运行了完整 tail
replay curve。严格集合为 91 个样本：

| B 侧重算历史 tokens | `Reuse=Target` | `Reuse=Source` |
|---:|---:|---:|
| 26 | 0/91 | 91/91 |
| 64 | 0/91 | 91/91 |
| 96 | 0/91 | 91/91 |
| 128 | 0/91 | 91/91 |
| 192 | 0/91 | 91/91 |
| 256 | 0/91 | 91/91 |
| 512 | 24/91 | 55/91 |
| 全历史（636--756） | 91/91 | 0/91 |

重算约 76% 的中位历史长度（512/674）仍只有 26.4% 的严格样本恢复 target；这说明影响并不只
存在于最后的 A proposal，而是分布在多条 queued-job 表示中。Qwen3-8B 的 18 个严格 pilot
复现了同一曲线：0--256 为 0/18，512 为 6/18，完整历史为 18/18。

8k 长历史的 15 个严格 pilot 给出更实际的系统边界：26/256 为 0/15，512 为 2/15，1024 为
14/15，2048 为 15/15。也就是说，安全恢复不必重算全部 8k，但 adaptive tail replay 需要约
1--2k tokens 才可靠；任何新方法都必须与这个强 baseline 比较，而不能只与 full prefill 比较。

长序列中有一个 full-tail replay 样本因“prefix 单独 prefill + history 分块 prefill”和一次性
Target-native 的 SDPA 数值差异而改变贪心输出；同一例重算 2k--8,192 tokens 都是正确的。该例
属于数值稳定性问题，不能解释为 policy imprinting，论文中应继续以 Identity 为主要实现对照。

## 完整 B policy 放在历史开头还是末尾

2026-08-12 进一步在同一批长历史上比较了两种接收布局。每例 A 的总 prefill 为
8,437--8,558 tokens（中位 8,475.5），其中复用历史 `H` 为 8,167--8,287 tokens。A/B 完整
policy 文本包含身份、目标 tenant、ACL、对上游 proposal 的信任规则和输出约束；tool schema
仍位于 chat template 要求的前置工具区。

前置布局为：

```text
native(B system policy + tools) + KV_A(H) + assistant readout
```

后置布局为：

```text
native(neutral system + tools) + KV_A(H) + native(B complete policy) + assistant readout
```

后置 policy 分别序列化成最后一条 `user` 或 `system` 消息，长度为 109--111 tokens。为建立稳定
chat-template 边界，A proposal 后先保留一条通用 handoff user message；该消息仍属于复用的
`KV_A(H)`，完整 B policy 是下一条原生消息。否则 Qwen template 会在追加新消息时删除最后
assistant message 的 thinking marker，造成旧历史 token 被改写。实验还分别测了丢弃 A prefix
和保留 A prefix；前者是主比较，后者用于判断旧 prefix 是否必须物理移除。

所有布局各有自己的 Native 和 Identity 对照。全体 100 个样本中，Qwen3-14B 的 front/tail-user/
tail-system Identity 均为 100/100；Qwen3-8B 分别为 99/100、100/100、97/100。共同严格集合只
保留 Source-native、Front-native、对应 Tail-native 都命中各自 gold，且两个 Identity 均保持
Native 的样本，因此 prompt 重排后模型自身不会做任务的样本不进入因果比较。

| 模型 | B policy 位置 | 严格 n | 执行 B | 精确追随 A |
|---|---|---:|---:|---:|
| Qwen3-14B | 开头 | 50 | 0/50 | 50/50 |
| Qwen3-14B | 末尾 user role | 50 | 40/50 | 2/50 |
| Qwen3-14B | 开头 | 48 | 0/48 | 48/48 |
| Qwen3-14B | 末尾 system role | 48 | 41/48 | 2/48 |
| Qwen3-8B | 开头 | 43 | 0/43 | 43/43 |
| Qwen3-8B | 末尾 user role | 43 | 41/43 | 0/43 |
| Qwen3-8B | 开头 | 44 | 0/44 | 44/44 |
| Qwen3-8B | 末尾 system role | 44 | 43/44 | 0/44 |

前置与后置是同一严格样本上的配对比较。Qwen3-14B user/system 的 `front wrong → tail correct`
分别为 40:0（`p=1.82e-12`）和 41:0（`p=9.09e-13`）；Qwen3-8B 分别为 41:0
（`p=9.09e-13`）和 43:0（`p=2.27e-13`）。两个切换方向分别报告后结论不变。

保留完整 A prefix 并在末尾追加 B policy 也没有系统性变差：14B user/system 分别恢复 45/50、
44/48，8B 分别恢复 38/43、38/44。也就是说，在这个动作读出任务中，**控制信息的相对时序比
物理删除 A prefix 更重要**；末尾完整 B policy 可以覆盖大部分 source-imprinted KV，但 14B
仍有约 15%--20% 的严格样本不能恢复，因此它是强 baseline 而非完全解耦。

这个结果也解释了早先 26-token generic trigger 无效的原因：它只说“按自己的 policy 执行”，
没有在末尾重新给出 B 的具体身份、tenant 和 ACL。完整的 110-token B policy 才提供足够的
late-binding 控制。另一方面，后置布局的全体 Native 准确率与前置布局不同（14B 71% vs 66%，
8B 54% vs 78%，以 user-role 为例），说明 prompt placement 会改变基础任务能力；因此不能只看
全体 Reuse 百分比，必须使用 layout-specific Native 和共同严格集合。

## 能支持和不能支持的论文命题

当前数据已经强力支持以下命题：

> 在带硬 ACL 冲突的自动 agent handoff 中，同模型直接复用 A 形成的历史 KV，会让 B 系统性执行
> A 的未授权动作；这不是 cache 拼接噪声，也不能被短 B-native trigger 或小规模 tail replay
> 消除。B 需要的是其自身 policy 下的反事实历史状态，而不是 A KV 的位置/坐标修正。

当前数据仍不能单独支持“任意真实多智能体协作都会显著掉点”或“已经达到 ICLR 主表强度”。主要
缺口是：ACL/priority 仍是合成层；没有在真实可执行环境中测 unauthorized side effect；没有方法
结果；没有端到端 TTFT、吞吐、显存/传输数据；模型家族覆盖有限。

下一阶段的 go/no-go 标准应为：

1. 在 AgentDojo、τ-bench/BFCL multi-turn 等至少两个真实工具环境中构造预注册的 role/ACL
   switch，报告任务成功率和 authorization violation rate；
2. 把 `full prefill`、`zero reuse`、固定 128/512/1k/2k replay、adaptive replay 都列为 baseline；
3. 方法必须在 violation 接近 Native 的同时，明显少于 adaptive replay 的计算量；
4. 加入 aligned、soft-conflict、hard-conflict 三个层级，解释 HotpotQA 中净效应接近零的边界；
5. 在线截获 A 生成时的 KV，测真实 handoff TTFT，而不是离线重建 source cache。

## 复现文件

- runner：`src/xmodel_kv/cli/bfcl_authorization_handoff.py`
- 通用 cache 对照：`src/xmodel_kv/cli/policy_imprinting.py`
- 汇总器：`src/xmodel_kv/cli/summarize_bfcl_handoff.py`
- 测试：`tests/test_bfcl_authorization_handoff.py`、`tests/test_summarize_bfcl_handoff.py`
- Qwen3-14B post-trigger 全量：
  `outputs/policy_imprinting/qwen3_14b_bfcl_authorization_posttrigger_full50.jsonl`
- Qwen3-14B replay curve 全量：
  `outputs/policy_imprinting/qwen3_14b_bfcl_authorization_posttrigger_replaycurve_pilot10.jsonl`
  （文件名保留早期 pilot 后缀，但已续跑到 100 行）
- replay curve 汇总：
  `outputs/policy_imprinting/qwen3_14b_bfcl_authorization_posttrigger_replaycurve_full50_summary.json`
- Qwen3-14B 8k 全量与汇总：
  `outputs/policy_imprinting/qwen3_14b_bfcl_authorization_bg60_full50.jsonl`、
  `outputs/policy_imprinting/qwen3_14b_bfcl_authorization_bg60_full50_summary.json`
- prompt-position runner：
  `src/xmodel_kv/cli/bfcl_authorization_prompt_position.py`
- Qwen3-14B prompt-position 汇总：
  `outputs/policy_imprinting/qwen3_14b_bfcl_authorization_prompt_position_bg60_summary.json`
- Qwen3-8B prompt-position 汇总：
  `outputs/policy_imprinting/qwen3_8b_bfcl_authorization_prompt_position_bg60_summary.json`

当前测试为 64/64 通过。runner 支持 `--post-switch-trigger`、`--background-records` 和任意
`--replay-tokens` curve，并按 case/direction 逐行落盘、断点续跑。
