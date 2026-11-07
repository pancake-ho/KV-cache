# Post-computation Tail-KV：MuSiQue 迭代结果（2026-08-17）

## 结论

本轮得到一个强正现象和一个明确失败边界：

1. **同模型、无训练的 post-answer Tail-KV handoff 明确成立。** Agent A 在约
   6.4k-token、多文档联合上下文上完成第一跳后，只截取实际生成答案对应的末尾
   K/V，去除源绝对位置 RoPE，再放到 Agent B 私有 system prefix 后重新施加 RoPE。
   gold-answer tail 在 64 个未见 MuSiQue dev case 上可被 Agent B 直接读回 64/64，
   并完成第二跳 63/64；真实 generated tail 完成第二跳 50/64，而 no-summary 只有
   6/64。
2. **INT4 不破坏主要信号。** per-token/per-head 对称 INT4 下，gold tail 仍可直接
   读回 64/64、第二跳 61/64；generated tail 为 29/64 direct bridge、46/64 第二跳。
3. **当前 learned codec 失败。** 将真实 Tail-KV 池化为 8 slots、经 MLP 压成
   `8 x 256` latent，再用 receiver manifold 物化 KV，在相同 held-out 64 case 上只有
   8--9/64 第二跳、0/64 direct bridge。即使 Qwen-to-Qwen 或只训练 bridge loss，
   direct bridge 仍为 0/64。

因此当前最准确的判定是：

> post-computation answer-tail KV 是一种可独立搬运、位置可重定向的短计算状态；
> 简单 INT4 可以稳健压缩它，但用小数据训练的池化 MLP codec 不能对开放词表实体
> 泛化。第一道“短状态存在性”门强通过，第二道“极小 payload / 跨模型 codec”门
> 未通过。

这比此前的 prefill-tail readout 更强，因为它不是从长 cache 末尾随机附加 readout
slots 并期待其主动搜索事实，而是复用模型已经完成计算、已经生成答案之后的真实
尾状态。文档可以联合编码、相互依赖、顺序任意；方法不要求 document-wise 独立 cache。

## 任务与严格切分

- Agent A：Qwen3-8B，在平均约 24k characters、6.4k tokens 的多文档 dossier 上回答
  MuSiQue 第一跳。
- Agent B：读取另一个 lookup 文档，根据第一跳实体完成第二跳。
- 训练 codec 使用 MuSiQue official train 构造的 128 case。
- 最终评估使用 official dev 构造的 index 32--95，共 64 case；该切片未参与此前
  dev 0--31 的结构和学习率调试。
- 128 train 与 64 eval 的 case ID、第一跳答案、第二跳答案和 `(bridge, final)` pair
  均无重叠。评估内部有 53 个唯一 pair（11 个重复 case），因此统计显著性仍应在
  更大的 entity-disjoint 集合上复验。

## 方法：无训练 Tail-KV transplant

设 Agent A 的完整 cache 为 `P_A + D_A + Y_A`，其中 `Y_A` 是已经生成的短答案。
只取所有层中 `Y_A` 对应的 K/V：

1. 对 K 使用 Qwen 自身 rotary schedule，按源绝对位置移除 RoPE；V 不变。
2. 将 content-space K 按 Agent B system prefix 后的新位置重新施加 RoPE。
3. 将重定位后的 Tail-KV 拼接到 Agent B 的原生 system cache 后。
4. Agent B 的 query 从 `prefix_length + tail_length` 继续正常 decode。

这一步没有重新 prefill `Y_A`，也不向 Agent B 暴露 answer token IDs。它只传末尾各层
的 K/V。量化版本对每个 layer/token/head 的 K 和 V 分别计算一个 fp16 scale，并做
对称 INT8/INT4 fake quantization；payload 统计包含 scale。

## 主要结果

### Gold answer-tail：机制上限

| Tail precision | direct bridge | 第二跳 | no-summary 第二跳 | mean tail tokens | mean payload |
|---:|---:|---:|---:|---:|---:|
| BF16 | **64/64** | **63/64** | 6/64 | 3.281 | 483,840 B |
| INT8 | **64/64** | **63/64** | 6/64 | 3.281 | 245,700 B |
| INT4 | **64/64** | **61/64** | 6/64 | 3.281 | 124,740 B |

BF16 唯一失败 case 的第一跳是 `Dennis Quaid`、目标第二跳是 `Meg Ryan`；接收模型
直接重复了 `Dennis Quaid`。这说明 Tail-KV 的 lexical identity 完整保留，但第二跳
任务仍可能发生 relation-following 错误。

### Generated answer-tail：真实 Agent A

| Tail precision | direct bridge | 第二跳 | 第二跳 F1 | no-summary 第二跳 | mean tail tokens | mean payload |
|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 29/64 | **50/64** | 0.789 | 6/64 | 6.391 | 942,336 B |
| INT8 | 29/64 | **49/64** | 0.784 | 6/64 | 6.391 | 478,530 B |
| INT4 | 29/64 | **46/64** | 0.737 | 6/64 | 6.391 | 242,946 B |

Agent A 生成答案 exact 为 29/64，contains 为 48/64。BF16 第二跳的条件统计为：

- source exact 时 28/29（96.6%）第二跳正确；
- source contains 时 46/48（95.8%）第二跳正确；
- source 不 contains 时只有 4/16（25%）正确。

因此 50/64 不是 Agent B 凭空猜测：主要增益由 Tail-KV 中存在的 Agent A 结果驱动；
少部分 source surface 不 exact 但包含答案或是足够相关的别名/短句，仍可完成第二跳。

### 相对完整 cache 的缩减

Qwen3-8B 每个 BF16 KV token 是 147,456 B。平均完整源 cache 约 899 MiB/case：

| 状态 | source/tail token ratio | BF16 full-cache / INT4-tail wire ratio |
|---:|---:|---:|
| gold tail | 1,948x | 7,556x |
| generated tail | 1,001x | 3,882x |

这里的 reduction 来自两个正交因素：只保留计算后的短 tail，以及 tail 内部 4-bit
量化。接收侧只需为 tail 分配 KV，而不是 6.4k-token 源 cache。

## Layer-selective Tail-KV：functional state 与 lexical state 分离

在 gold answer-tail 上固定 INT4，只发送前 K 层的 tail K/V；未发送层由接收端补零，
所有层仍保留相同的逻辑 tail positions。结果如下：

| transmitted prefix layers | 第二跳 | direct bridge | 第二跳 F1 | mean payload |
|---:|---:|---:|---:|---:|
| 6/36 | 7/64 | 0/64 | 0.163 | 20,790 B |
| 9/36 | 12/64 | 0/64 | 0.233 | 31,185 B |
| 12/36 | 32/64 | 0/64 | 0.530 | 41,580 B |
| 15/36 | 51/64 | 0/64 | 0.803 | 51,975 B |
| 18/36 | 58/64 | 0/64 | 0.927 | 62,370 B |
| 21/36 | 60/64 | 0/64 | 0.948 | 72,765 B |
| 24/36 | **61/64** | 0/64 | 0.953 | 83,160 B |
| 27/36 | **61/64** | 1/64 | 0.953 | 93,555 B |
| 30/36 | 60/64 | 17/64 | 0.938 | 103,950 B |
| 33/36 | 60/64 | 51/64 | 0.938 | 114,345 B |
| 36/36 | **61/64** | **64/64** | 0.953 | 124,740 B |

这产生了清晰的两条恢复曲线：

- **functional state**：前 15--24 层已经足够驱动大多数第二跳任务；
- **lexical state**：逐字读回第一跳答案直到约 30 层才开始恢复，接近全 36 层才完整。

简单的“早层只是词面、晚层只是语义”解释与结果相反。更合理的机理是：Agent B 的
query hidden states 在早期层通过 tail K/V 获得足以继续关系推理的条件信号，此后即使
没有 tail，它们仍能在后续层传播；而 direct generation 每一步都需要更完整的深层
lexical trace 才能稳定输出实体表面形式。

在真实 generated tail 上复验：

| transmitted prefix layers | 第二跳 | direct bridge | mean payload |
|---:|---:|---:|---:|
| 12/36 | 19/64 | 0/64 | 80,982 B |
| 15/36 | 40/64 | 0/64 | 101,228 B |
| 18/36 | 44/64 | 0/64 | 121,473 B |
| 21/36 | **48/64** | 0/64 | 141,718 B |
| 24/36 | 46/64 | 0/64 | 161,964 B |
| 36/36 | 46/64 | 29/64 | 242,946 B |

前 21 层相对全层为 3 个 case 新增命中、1 个 case 丢失，paired exact McNemar
`p=0.625`；不能声称它提高准确率，但可以说在此样本上以 **41.7% 更少 wire bytes**
保持了统计不可区分的第二跳质量。source contains 时前 21 层为 44/48（91.7%），
source 不 contains 时为 4/16，仍然主要由 Agent A state 驱动。

控制性 layer pattern 也支持深度顺序的重要性。gold INT4 下，只发前 18 层是 58/64，
只发后 18 层是 8/64；交错奇数层是 39/64，交错偶数层是 7/64。这不是任意 18 层都
等价。需要注意，未发送层补零不是最优重建，因而这些是结构性 ablation，而非最终
layer-drop codec。一个实际系统需要隐式零槽 kernel 或从已发送层低成本重建未发送层。

这一结果给论文带来比“短 tail”更强的可检验主张：**KV handoff 的任务可用性与明文
可恢复性可以被层选择解耦。** 它也提供了避开“直接传 3 个文本 token 更便宜”质疑的
方向：评估 functional state，而不是把 exact lexical reconstruction 当作唯一目标。

### KV-head ablation：固定稀疏删除不可取

在 first-21 / INT4 sweet spot 上继续删除 Qwen 的 8 个 KV heads：

| transmitted heads in first 21 layers | gold 第二跳 | generated 第二跳 | generated payload |
|---:|---:|---:|---:|
| all 8 | **60/64** | **48/64** | 141,718 B |
| first 6 | 49/64 | 39/64 | 106,289 B |
| last 6 | 39/64 | 31/64 | 106,289 B |
| even 4 | 25/64 | 23/64 | 70,859 B |
| odd 4 | 14/64 | 8/64 | 70,859 B |
| first 4 | 28/64 | -- | -- |
| last 4 | 5/64 | -- | -- |

head 重要性明显不均匀（first/even 优于 last/odd），但即使删掉相对较弱的两个 heads，
准确率下降也很大。与 first-K layer truncation 不同，固定 head sparsity 没有找到近似
accuracy-preserving 的 operating point。因此下一步应保持所有 KV-head 的覆盖，通过
head 内低秩、跨层共享 basis 或 residual reconstruction 压缩，而不是把某些 heads
直接置零。

### Per-token head-rank：有效秩并不低

对 first-21 的每个 layer/token，将 K 和 V 分别看作 `8 KV heads x 128 channels`
矩阵，做逐样本 truncated SVD。factor 在 BF16 或 INT4 下传输并在接收端重建：

| head rank | BF16 factor 第二跳 | INT4 factor 第二跳 | BF16 payload | INT4 payload |
|---:|---:|---:|---:|---:|
| full 8 | **63/64** | **60/64** | 282,240 B | 72,765 B |
| 6 | 47/64 | 26/64 | 224,910 B | 60,086 B |
| 4 | 6/64 | 5/64 | 149,940 B | 40,792 B |
| 2 | 5/64 | 5/64 | 74,970 B | 21,499 B |

在全部 64 个 gold case、所有 layer/token/K/V 矩阵上，top-2/4/6 singular components
平均只保留 48.0% / 72.5% / 89.2% Frobenius energy；rank-6 的 case-level mean energy
范围也只有 88.8%--89.6%。这与功能曲线一致：该轴上的 tail KV 本来就是高秩，
rank-6 并非小误差近似。INT4 factor 又与 rank error 叠加，使 47/64 进一步降到 26/64。

因此“先验证 delta/tail rank”在这里给出了明确否定：**不能依赖逐 token 的
head-channel low rank。** 后续若做低秩，应尝试跨 token、跨相邻 layer 的共享结构或
相对一个可重建 baseline 的 residual rank，而不是直接分解原始 `8x128` 矩阵。

## Learned codec 的负结果

### Receiver-manifold 与真实 Tail-KV encoder

encoder 从 Qwen 的 early/mid/final 三层真实 answer-tail K/V 中移除 RoPE，pool 到 8
slots，经 MLP 得到 `8 x 256` latent；decoder 将 latent 映射为 Llama 或 Qwen 的软
embeddings，再由冻结 receiver Transformer 物化原生 KV。128 train case 的驻留
特征仅 24 MiB，训练过程稳定。

| Source -> receiver / objective | source state | FP16 第二跳 | direct bridge | no-summary |
|---:|---:|---:|---:|---:|
| Qwen -> Llama / joint | generated | 9/64 | 0/64 | 7/64 |
| Qwen -> Llama / joint | gold | 8/64 | 0/64 | 7/64 |
| Qwen -> Llama / bridge-only | gold | 8/64 | 0/64 | 7/64 |
| Qwen -> Qwen / joint | gold | 8/64 | 0/64 | 6/64 |
| Qwen -> Qwen / bridge-only | gold | 9/64 | 0/64 | 6/64 |

联合 Qwen->Llama 的小幅增益不显著：generated 相对 no-summary 的 discordant cases
为 6 胜/4 负，exact McNemar `p=0.754`；gold 为 7 胜/6 负，`p=1.0`。bridge-only
训练 CE 可以降至 1.19，但 held-out direct bridge 仍为零，说明不是多任务 loss 冲突，
而是对 128 个训练实体的记忆，没有学会开放词表 identity transport。Qwen-to-Qwen
同样失败，说明瓶颈也不能简化为“跨模型结构不同”。

对比 raw transplant 的 64/64 direct bridge，这一控制把失败位置明确定位在：

- early/mid/final 取样与固定池化破坏了逐层 lexical trace；
- 小 MLP 必须从 128 个实体中学习近似 vocab inversion，不可能开放词表泛化；
- receiver soft-manifold 改善了优化稳定性，但没有提供结构性的 token identity path。

## Pre-answer state：天然 tail 不成立，学习式 state emission 成立

### 原始 pre-answer tail 是负结果

为了排除 answer token 只是把明文重新编码进 KV，首先在 Agent A 尚未输出答案时，直接
截取 assistant generation header 前的 1/4/8 个原始 KV。64 个 case 上，BF16 第二跳
分别只有 10/12/11，no-summary 为 6；4-token correct-source 为 12/64，而 circularly
shifted source 反而为 13/64。相反，post-answer correct-source 为 50/64，shifted-source
仅 8/64。这说明普通 prompt tail 在答案形成前没有可复用的因果状态，强信号确实是在
答案计算/生成期间写入 tail，而不是任意末尾 KV 都有效。

### Contextual state-emission capsule

随后实现了不生成答案文本的 4/8-slot soft capsule。共享的 learned embeddings 被附加在
`dossier + Agent-A question + assistant header` 之后；它们通过冻结 Qwen 的真实 36 层
计算路径 attend 完整 source prefix，每个 case 因而物化出不同的原生逐层 K/V。传输时
丢弃约 6.3k-token source cache，只保留这些 slots；Key 先逆 source RoPE，再施加 receiver
system-prefix 后的新 RoPE，Value 不变。base model 完全冻结，只训练 slot embeddings。

这与失败的 dense codec 有两个结构差别：

- 它不从固定 latent 凭空反演每一层 K/V，而是沿 base model 的原生层间动力学生成；
- slot 的 case-specific 内容来自对完整 source cache 的注意力，而不是从 128 个实体中
  学一个封闭词表分类器。

训练目标同时包含第二跳 `stage CE` 和直接读回第一跳实体的 `bridge CE`。48-shot pilot
在调参切片上曾达到 13/32，但 fresh 64 例只有 11/64（BF16），相对 no-summary 8/64 的
McNemar `p=0.607`，F1 delta 95% CI `[-0.033, 0.178]`；同一 checkpoint 在训练集却是
47/48 stage、39/48 bridge。这一诊断确认小数据 soft prompt 严重过拟合，不能把 pilot
当作证据。

训练扩大到全部 128 cases 后，在独立 `dev[64:96]` 比较三个配置：

| validation 配置 | stage | bridge | no-summary |
|---|---:|---:|---:|
| 4 slots, stage:bridge = 1:1 | 6/32 | 3/32 | 3/32 |
| 8 slots, stage:bridge = 1:1 | 9/32 | 0/32 | 3/32 |
| **4 slots, stage:bridge = 1:2** | **12/32** | **10/32** | 3/32 |

预先按 validation 选定 4-slot bridge-heavy checkpoint 后，仅打开一次封存的
`dev[160:197]`：

| sealed test | stage | bridge | mean F1 | payload |
|---|---:|---:|---:|---:|
| no-summary | 6/37 | -- | 0.238 | 0 |
| capsule BF16 | 14/37 | 11/37 | 0.441 | 589,824 B |
| **capsule INT4** | **17/37** | 11/37 | **0.538** | **152,064 B** |
| generated-answer Tail-KV BF16 | 30/37 | 22/37 | 0.822 | 793,074 B |
| generated-answer Tail-KV INT4 | 29/37 | 22/37 | 0.795 | 204,464 B |
| gold-answer Tail-KV INT4 | 35/37 | 37/37 | 0.957 | 122,268 B |

sealed INT4 capsule 相对 no-summary 为 12 个独有命中、1 个反向命中，exact McNemar
`p=0.00342`；mean-F1 delta 为 0.300，paired bootstrap 95% CI `[0.154, 0.451]`。
BF16 也显著（10 胜/2 负，`p=0.0386`）。INT4 在这个 37-case split 高于 BF16，但在
下面的 64-case replication 低于 BF16，因此不能声称量化本身提升质量，只能说 4-bit
仍保留了显著信号。

在较大的 secondary `dev[96:160]` 上复验：BF16 capsule 为 24/64，INT4 为 17/64，
no-summary 为 8/64；相对 no-summary 的 exact McNemar 分别为 `p=8.55e-4` 和
`p=0.0490`。两个必要控制进一步定位了因果性：

| secondary control | BF16 stage / bridge | INT4 stage / bridge |
|---|---:|---:|
| 正确 source capsule | **24/64 / 20/64** | **17/64 / 18/64** |
| circular shift-1 source | 2/64 / 0/64 | 3/64 / 0/64 |
| 未训练的相同 4 slots | 7/64 / 0/64 | 8/64 / 0/64 |

正确 source 相对 shift-source 的 paired exact `p=2.98e-6`（BF16）和
`p=5.19e-4`（INT4）；训练后相对未训练 slots 为 `p=4.88e-4` 和 `p=0.0352`。
因此收益不是 target relation prior、任意额外 cache positions 或 initialization text，
而是 learned slots 从对应 source 中发射出的 contextual state。

### Learned capsule 的 layer compression

对 secondary 64 cases 扫描 INT4 capsule 的 first-K layers，未传层保留零槽：

| transmitted layers | stage | direct bridge | payload |
|---:|---:|---:|---:|
| 6 | 9/64 | 0/64 | 25,344 B |
| 12 | 4/64 | 0/64 | 50,688 B |
| 15 | 5/64 | 0/64 | 63,360 B |
| 18 | 7/64 | 0/64 | 76,032 B |
| 21 | 16/64 | 0/64 | 88,704 B |
| 24 | 14/64 | 0/64 | 101,376 B |
| **27** | **17/64** | 7/64 | **114,048 B** |
| 30 | 17/64 | 7/64 | 126,720 B |
| 33 | 16/64 | 13/64 | 139,392 B |
| 36 | **17/64** | **18/64** | 152,064 B |

first-27 与 all-36 的 64 个 exact correctness **逐例完全相同**；仅一个非 exact case
F1 下降，mean delta 为 -0.0078。于是 4 slots + first 27 layers 在保持 task accuracy
的同时，相对全 capsule 再省 25% bytes，相对同切片 generated-answer INT4 Tail-KV
平均省约 46%。与此同时 lexical bridge 仍需要更深层，再次表明 functional state 与
逐字可恢复性可以被层选择解耦。这个 operating point 还需在新 split 上预注册复验。

### Future-state behavioral distillation：负结果

一个自然想法是用几乎完美的 gold-answer Tail-KV 当 teacher，但不直接回归其高秩 K/V；
而是在相同 stage/bridge query 和 teacher-forced answer positions 上匹配完整词表 logits，
即 `KL(teacher || capsule)`。KL 按 answer position 平均、temperature=2；其它设置固定为
4 slots、bridge weight 2、N=128。

| teacher KL weight | validation stage | bridge | no-summary |
|---:|---:|---:|---:|
| 0（CE-only） | **12/32** | **10/32** | 3/32 |
| 0.01 | 7/32 | 10/32 | 3/32 |
| 0.03 | 6/32 | 8/32 | 3/32 |
| 0.10 | 6/32 | 10/32 | 3/32 |

teacher KL 从首步约 57 降至训练均值 5--7，说明它不是无法优化；但 KL=0.1 相对 CE-only
为逐例 0 个新增、6 个丢失，`p=0.03125`。更合理的解释是 post-answer teacher 的输出
分布依赖已经出现的 lexical/copy state，强迫 pre-answer slots 模仿这个“未来已知答案”
的状态，与学习 task-sufficient extraction 冲突。因此不继续扫描更大 KL 权重，也不把
raw KV/value regression 当作默认方向。

### Sender-only low-rank write adapter：正结果

soft-only 的训练/测试差距说明 4 个共享 embeddings 的 source-reading capacity 是瓶颈。
为此只在 capsule 物化期间，在 decoder layer 之间加入 sender-only low-rank residual：

`h_(l+1) <- h_(l+1) + W_up SiLU(W_down RMSNorm(h_(l+1)))`。

adapter 在 source prefill 时不启用，在 receiver generation 前移除；`k_proj/v_proj` 始终
使用冻结 Qwen 原始权重，传出的仍是 receiver-native KV，而不是 dense decoder 合成的
任意 cache。最后一层不加 block，因为它不会影响任何后续 cached K/V。rank-4 总计约
1.15M sender 参数，网络 payload 不变。

固定 validation 的 rank scan：

| writer | validation stage | bridge |
|---|---:|---:|
| soft-only | 12/32 | 10/32 |
| rank 1 | 11/32 | 11/32 |
| rank 2 | 10/32 | 11/32 |
| **rank 4** | **17/32** | 10/32 |

rank-4 相对 soft-only 是 8 个新增、3 个丢失；32 例 exact McNemar `p=0.227`，因此单独
看 validation 尚不能证明提升。在 secondary 64 例上，结果更清晰：

| secondary | BF16 stage / bridge | INT4 stage / bridge |
|---|---:|---:|
| soft-only | 24/64 / 20/64 | 17/64 / 18/64 |
| **rank-4 writer** | **31/64 / 22/64** | **30/64 / 23/64** |
| writer + shift-1 source | 2/64 / 0/64 | 3/64 / 0/64 |
| no-summary | 8/64 / -- | 8/64 / -- |

INT4 writer 相对 soft-only 为 18 个新增、5 个丢失，exact `p=0.0106`，mean-F1 delta
0.216、95% CI `[0.098, 0.336]`。正确 source 相对 shift-source 为 27 个新增、0 个丢失，
`p=1.49e-8`，说明 adapter 增强的是 source-conditioned state writing，而不是更强的
target prior。BF16 exact 提升尚不显著（`p=0.210`），但 F1 delta CI `[0.006, 0.250]`。

writer 也改变了 layer operating point：

| transmitted writer layers (INT4) | stage | bridge | payload |
|---:|---:|---:|---:|
| 18 | 11/64 | 0/64 | 76,032 B |
| 21 | 22/64 | 0/64 | 88,704 B |
| 24 | 27/64 | 1/64 | 101,376 B |
| 27 | 26/64 | 4/64 | 114,048 B |
| **30** | **29/64** | 10/64 | **126,720 B** |
| 33 | 29/64 | 17/64 | 139,392 B |
| 36 | 30/64 | 23/64 | 152,064 B |

first-30 相对 all-36 为 1 个新增、2 个丢失，McNemar `p=1.0`，F1 delta CI
`[-0.078, 0.016]`；它以 16.7% 更少 bytes 保持统计不可区分的 task quality。其 correct
source 29/64 对 shift-source 3/64（26 新增/0 反向，`p=2.98e-8`）。当前最佳 exploratory
点因此是 **4 slots + rank-4 sender writer + first-30 layers + INT4 = 126,720 B**。它比
同切片 generated-answer INT4 Tail-KV 平均小约 40%，但准确率仍是 29/64 对 52/64。
由于 secondary split 之前已用于 soft-only 分析，这一提升仍必须在全新 seed/benchmark
上预注册复验，不能当作新的 sealed-test 数字。

### 固定配置的 256-case ID-disjoint test

为避免继续复用 197-case dev，benchmark builder 增加 `--exclude-case-ids-from`。从
MuSiQue train 的 1,639 个 eligible cases 中显式排除 writer 训练的 128 个 case IDs，
再以 seed 2031 一次性冻结 256 个新问题；配置在运行前固定为 rank-4 writer、4 slots、
first-30、INT4。正确 source、circular shift-1 和 generated-answer Tail-KV 同时启动，
不以中途结果停止或改参。

| frozen 256-case test | stage | bridge | mean F1 | payload |
|---|---:|---:|---:|---:|
| no-summary | 42/256 (16.4%) | -- | 0.282 | 0 |
| shift-source writer | 48/256 (18.8%) | 2/256 | 0.303 | 126,720 B |
| **correct-source writer** | **178/256 (69.5%)** | **96/256** | **0.757** | **126,720 B** |
| generated-answer Tail-KV | 217/256 (84.8%) | 120/256 | 0.864 | 194,090 B mean |

writer 相对 no-summary 为 147 个独有命中、11 个反向，McNemar
`p=1.59e-31`；mean-F1 delta 0.475，按 108 个 unique bridge entities 整簇 bootstrap
的 95% CI 为 `[0.400, 0.548]`。正确 source 相对 shift 为 134 个独有命中、4 个反向，
`p=8.55e-35`；clustered F1 CI `[0.363, 0.543]`。因此大样本因果结论不依赖重复实体。

writer 相对 generated Tail-KV 少 34.7% bytes，并避免先自回归生成平均 5.1 个答案 tokens，
但 accuracy 低 15.2 points（27 writer-only、66 Tail-only，`p=6.47e-5`）。两者互补：
151 个都对、12 个都错，oracle union 为 244/256（95.3%）。这表明 pre-answer writer
不是 post-answer Tail-KV 的简单弱复制，而能修复一部分生成错误。

尝试了运行时可观测的单标量 gate：在旧 secondary 上分别用 generated answer 的 mean
log-prob、minimum log-prob、mean top1--top2 margin 选阈值，规则为低置信度走 writer、
高置信度走 Tail-KV。三个 calibration 的最优解都是 `threshold=-inf`，即始终走 Tail；
因此没有在 256-case test 上事后调阈值。互补性真实，但普通 token confidence 尚不足以
路由，后续需要学习式 verifier 或 receiver-side state agreement，而非宣称已有 ensemble。

需要严格区分：这些 256 个 **case IDs 未见**，但它们与 128 training cases 都来自原始
MuSiQue train split，且同 relation decoy pool 会共享非 supporting distractor 分布。因此
它是强的 ID-disjoint / same-source-distribution 复验，不等价于新的原始 dataset split。
旧 MuSiQue dev 的正结果提供 split-shift 证据，下一篇正式实验还必须加入另一 benchmark
或重新构造完全隔离的 source pools。

### 跨数据集零微调：HotpotQA 200-case frozen test

随后在运行前冻结协议，将同一个 MuSiQue-trained checkpoint 原样用于 LongBench HotpotQA
全部 200 题；没有任何 HotpotQA gradient update。source 平均 13,458 tokens，Agent A
完整上下文回答本身为 42.0% EM / 0.570 F1。两种 receiver protocol 的 pooled 结果为：

| zero-shot HotpotQA | no summary | shifted capsule | correct capsule | generated Tail-KV |
|---|---:|---:|---:|---:|
| state readout EM / F1 | 0.0 / .002 | 0.0 / .003 | **10.5 / .169** | 34.5 / .503 |
| question-conditioned EM / F1 | 16.5 / .274 | 13.5 / .213 | **21.5 / .309** | 41.5 / .560 |

state readout 中 correct capsule 相对 shift 为 21 个新增、0 个反向，exact
`p=9.54e-7`；F1 delta `+0.166`，95% CI `[+0.122,+0.212]`。question-conditioned
中 correct 相对 shift 为 19:3，`p=8.55e-4`；F1 delta `+0.096`，CI
`[+0.051,+0.144]`。因此 **跨数据集 source-conditioned state emission 门通过**。

但 question-conditioned capsule 相对 no-summary 虽然 EM +5.0 points（14:4，
`p=0.0309`），F1 delta `+0.035` 的 CI 为 `[-0.004,+0.077]`，所以严格的净效用
门不能判为完全通过。generated Tail-KV 仍显著更强。capsule 126,720 B，相比 all-36
INT4 generated Tail-KV 的平均 206,617 B 少 38.7%。完整协议、系统 profile 和失败边界见
`docs/hotpot_zeroshot_capsule_results_20260817.md`。

### late-layer packet：独立 290-case confirmatory replication

standard-200 上的 post-hoc 2×2 诊断表明 INT4 不是主要损失，first-30 丢弃的最后 6 层
破坏 OOD state。进一步探索得到 `last18 + INT4 = 76,032 B`。由于这是看过 standard
结果后选择的配置，另从 HotpotQA-E 300 题中排除与 standard 重复的 10 个 normalized
questions，冻结剩余 290 题、配置、三分片和判定门后再运行。

| question-disjoint E290 | no summary | shifted last18 | correct last18 | generated Tail-KV |
|---|---:|---:|---:|---:|
| state readout EM / F1 | 0.0 / .000 | 0.34 / .009 | **11.7 / .234** | 35.2 / .524 |
| question-conditioned EM / F1 | 19.7 / .282 | 15.2 / .211 | **31.0 / .437** | 42.8 / .581 |

预注册的 question-conditioned 两个 F1 门均通过：correct-vs-no delta `+.156`，CI
`[+.109,+.203]`；correct-vs-shift delta `+.226`，CI `[+.177,+.275]`。state-readout
correct-vs-shift delta `+.225`，CI `[+.184,+.266]`。三个执行 shard 的方向一致。

last18 payload 相比 generated Tail-KV 平均 200,436 B 少 62.1%；共同 source prefill 后，
同步单请求 handoff 为 222 ms 对 441 ms，少 49.5%，但 capsule F1 仍低 .144，属于明确的
quality--latency--payload tradeoff，不是等质量加速。完整结果见
`docs/hotpot_last18_confirm_results_20260817.md`。

### Llama-3.1-8B：无搜索的第二模型族复现

完全复制 Qwen writer 的训练超参，在 Llama-3.1-8B 上训练 4-slot/rank-4 writer；固定
MuSiQue dev32 上，downstream EM/F1 从 no-summary 的 21.9%/.266 提升到 40.6%/.444。
按事前冻结的 architecture-normalized `last16 + INT4`，在同一个 HotpotQA-E 290 题上
直接运行，没有 Llama layer sweep。

| Llama E290 | no summary | shifted last16 | correct last16 | generated Tail-KV |
|---|---:|---:|---:|---:|
| state readout EM / F1 | 0.0 / .007 | 0.0 / .013 | 0.34 / .016 | 1.38 / .148 |
| question-conditioned EM / F1 | 10.7 / .195 | 8.62 / .154 | **15.9 / .304** | 21.4 / .310 |

question-conditioned correct-vs-no F1 为 `+.109`，CI `[+.074,+.145]`；correct-vs-shift
为 `+.150`，CI `[+.113,+.187]`。因此 source-conditioned task utility 在第二模型族
无调参复现。与此同时，generic state readout 完全没有复现，说明 late-layer capsule
不是通用可朗读文本容器，而是被下游 query 激活的 continuation state。

67,584 B capsule 比 Tail-KV 平均 175,369 B 少 61.5%，post-prefill handoff 为 253 ms
对 400 ms，少 36.7%。Tail 的 EM 显著更高，但两者 F1 差仅 `.006`，paired CI
`[-.039,+.051]`，当前样本无法分辨。完整协议与结果见
`docs/llama31_writer_replication_preregister_20260817.md` 和
`docs/llama31_writer_replication_results_20260817.md`。

### Llama 数据扩展与深度定位：四个 slots 是 depth-indexed state

随后把 Llama writer 的训练集从 128 扩到 512，保持 4 slots、rank-4、1600 steps
不变，并在同一旧 test256 上做 all/first16/last16 与 BF16/INT4 因子诊断：

| writer | all BF16 F1 | all INT4 F1 | first16 INT4 F1 | last16 INT4 F1 |
|---:|---:|---:|---:|---:|
| N=128 | .760 | .750 | .536 | .551 |
| N=512 | **.859** | **.829** | **.643** | .349 |

N512 相对 N128 的 all-BF16 F1 增加 `.099`，但 last16-INT4 反而下降 `.202`；两者
paired bootstrap CI 都排除零。增加数据确实让完整逐层 state 更好，却把信息分布到更
多深度，不能推导出任意固定 half-depth packet 也会更好。INT4 相对 BF16 的损失远小于
layer truncation，因此主要瓶颈是注入拓扑而不是 4-bit 量化。

在看 first16 结果之前冻结的新 `confirm256`（排除全部 train512 和旧 test256，共 768
IDs；99 个 bridge clusters）上，结果为：

| confirm256 arm | EM | F1 | payload |
|---|---:|---:|---:|
| no-summary | .148 | .255 | 0 |
| shifted first16 | .152 | .252 | 67,584 B |
| correct last16 | .270 | .362 | 67,584 B |
| **correct first16** | **.570** | **.677** | **67,584 B** |
| **correct all-layer** | **.812** | **.880** | **135,168 B** |

first16 相对 no、shift、last16 的 F1 delta 分别为 `+.422`、`+.425`、`+.314`；普通
bootstrap 和按 bridge cluster 整簇 bootstrap 的 95% CI 下界都大于零。相同 67.6 KB
payload 的 first16 与 last16 相差 31.4 F1 points，说明 capsule 不是“四个可互换向量”，
而是四个在每个深度承担不同功能的 state positions。first16 是早期 functional
injection 通道；all-layer 继续保留深层 lexical/readout state，构成 67/135 KB 两个
明确 Pareto 点。

一个预注册的 hard-budget 训练负结果进一步支持“拓扑而非单纯优化不足”：直接让 N512
writer 在训练中只看到 last16+INT4（STE）后，dev32 从 unconstrained writer 的 `.266`
提高到 `.315` F1，但 delta `.049` 的 CI 穿零，未达到预设 `+.10` 门；恢复其未约束的
早层 K/V 后 all-layer F1 又降到零。固定 hard mask 可优化，但未发送层若没有 null/多预算
约束会成为干扰源。后续应做 stochastic multi-budget 或 budget-conditioned routing，
不应继续假定 last-half 是通用 packet。

### 实际 INT4 packet 与 2WikiMQA 零微调迁移

量化路径已从 fake quantization 补成实际序列化 codec：signed 4-bit nibbles、FP16
per-layer/token/head scales、layer indices 和 CRC。Llama 4-slot first16 的 accounting 为
67,584 B，完整 framed packet 为 67,642 B。N128 test256 上 real-codec F1 `.554`，fake
INT4 为 `.551`，paired delta `.0039`、CI `[-.0078,.0195]`；H200 reference 的 pack/unpack
分别约 4.65/2.71 ms。因而“4-bit 有用”不再只由浮点模拟支持，但仍缺 serving fused
kernel 和真实链路测量。

同一个 MuSiQue N512 checkpoint 随后原样用于从未训练、未选层的 LongBench
2WikiMQA 全部 200 题。first16 的两个预注册 question-conditioned gate 都通过：

| 2Wiki question-conditioned | EM | F1 | payload |
|---|---:|---:|---:|
| no-summary | .045 | .141 | 0 |
| shifted first16 | .150 | .207 | 67,584 B |
| generated-answer Tail-KV | .115 | .208 | 184,673 B mean |
| all-layer capsule | .130 | .256 | 135,168 B |
| **first16 capsule** | **.200** | **.267** | **67,584 B** |

first16-vs-no F1 delta `+.126`，CI `[+.079,+.176]`；first16-vs-shift 为 `+.060`，
CI `[+.013,+.107]`。在 source-answer exact 的 86 题上，first16 为 `.424` F1，generated
Tail 为 `.248`，差值 CI 也排除零。all-layer 在 generic state readout 达到 `.125` F1，
first16 只有 `.022`，但 question-conditioned F1 没有提高（`.256` vs `.267`）。这再次
分离了早层任务注入与深层词汇读出。

由于 first16/all-layer 来自独立执行，另事前冻结一次完整 first16 repeat。两种协议下
200/200 个 capsule 输出、EM、F1 均与原始执行逐字符串完全一致，差值 bootstrap CI
为 `[0,0]`；no-summary 跨执行也 200/200 一致，而 generated Tail 有 25/200 个字符串
变化。该复现门通过，证明选择的 KV path 稳定；生成答案的表面随机性应单独报告。

### 多预算 writer：两个有信息量的负结果

在排除既有 1,024 个 MuSiQue ID 的 budgetdev128 上，固定一个 4-slot writer 随机面对
`first_16` 与 `all` 两种接收预算。INT4-aware all-only control 与历史 BF16-trained writer
无可辨别差异（first16 F1 `.691`，all `.841`），因此 STE 本身不是瓶颈。

朴素多预算训练在两种预算都使用 stage+lexical bridge loss，first16/all F1 降至
`.591/.717`；相对 all-only 的 paired delta 为 `-.100/-.124`，两道质量门均失败，但正确
source 相对 shift-1 的 first16 delta `+.334`，说明它仍是因果状态而非固定提示。

预注册仅允许一个机制修正：first16 update 的 bridge weight 设为 0，all update 设为 4，
保持相同期望权重和 801/799 次预算采样。这个 functional/lexical 版本进一步降至
`.553/.728`。相对 all-only：

- first16 F1 delta `-.1375`，95% CI `[-.2273,-.0492]`；
- all F1 delta `-.1135`，95% CI `[-.1992,-.0307]`；
- 正确 source 相对 shift-1 的 first16 delta `+.2523`，CI `[+.1633,+.3422]`。

cluster bootstrap 给出相同结论。因此冲突并不只是“浅层不应恢复词面”：不同深度接收
拓扑对同一组发射 K/V 的最优梯度仍会竞争。一个固定小 packet 不能被默认视为更大 packet
的最优前缀。budgetdev128 已按约束停止选型；下一版本必须改变表示本身，而不是继续扫描
loss weight。

### 新独立确认：硬 base/residual 隔离仍失败，但普通 full-depth 数据扩展显著提升 compact packet

在前述 1,152 个已用 ID 之外，新冻结 256 条增量训练 case；随后再排除它们，冻结独立
confirm128（73 个 bridge-answer clusters）。两条训练均从同一个 N512 all-only parent
初始化、使用 3,200 次 update：control 继续训练 all-layer；progressive candidate 在 layer16
同时切断 sender hidden 和 receiver early-KV 梯度，使 first16 base 与 deep residual 分治。

progressive 的三道预注册门为：

- first16 相对 compute-matched control `-.1531` F1，CI `[-.2344,-.0742]`；
- all 相对 control `-.1409`，CI `[-.2122,-.0724]`，非劣门失败；
- correct first16 相对 shift-1 `+.4209`，CI `[+.3323,+.5084]`，因果门通过。

cluster bootstrap 完全同结论。硬梯度边界甚至使 progressive all 比 parent 低 `.0859` F1，
CI 排除零。早/深层虽然承担不同功能，却不是可独立训练的两个模块：deep receiver 的 loss
需要通过完整 sender graph 给 early embeddings/adapter 有用的 credit。

同一实验产生了更简单的正结果。普通 all-layer continuation 使用新增 256 条数据后，
confirm128 first16 从 parent 的 `.734` 提高到 `.832` F1，delta `+.0984`，普通 CI
`[+.0359,+.1641]`、cluster CI `[+.0378,+.1621]`；all 从 `.819` 提高到 `.874`，但该 delta
区间穿零。这说明当前经过验证的优化不是 compact-specific training，而是 full-depth
端到端训练后在 serving 时截取 early depth prefix。

### 2WikiMQA-E300 密封确认：compact state 与因果性成立，compact scaling 未成立

在读取文件内容前冻结协议后，使用完整且此前未触碰的 `2wikimqa_e` 300 题恰好执行
三个 arm：continued checkpoint first16、parent first16、continued all-layer。四个 slots、
真实 INT4、prompt、24-token decode 和 source shift 全部固定。

question-conditioned 绝对结果为：no-summary `.1545` F1、generated Tail `.2142--.2168`、
parent first16 `.2813`、new shifted first16 `.2533`、new first16 `.3007`、new all `.3798`。
预注册三道门中：

- new first16 相对 no-summary `+.1462`，CI `[+.1014,+.1917]`，通过；
- new first16 相对 shift `+.0474`，CI `[+.0041,+.0911]`，通过；
- new first16 相对 parent 仅 `+.0194`，CI `[-.0200,+.0591]`，既略低于 `+.02` 点估计门，
  CI 也穿零，失败。

因此密封集强确认的是 **67,584 B 的 compact、sample-specific summary KV 存在**，而不是
“新增 MuSiQue 数据显著提高 compact 跨任务泛化”。all-layer 相对 first16 增加 `.0791`
question F1（CI `[+.0295,+.1302]`），state-readout 增加 `.1450`（CI
`[+.1117,+.1799]`），同时 payload 翻倍到 135,168 B。all correct 相对 shift 的 question
F1 更高 `.2431`，说明深层恢复的是强样本绑定的 lexical/readout state，而非通用提示先验。

这一结果把优化问题进一步收窄：full-depth 联合训练是必要的，但普通数据扩展会把新增
可迁移信息更多写入深层。下一步不是再扫描 hard split，而是设计保持全图 credit assignment
的 early-bottleneck / deep-residual 可压缩性约束。

### 连续 source-read bottleneck：all-layer 可恢复，但 hidden/KV 边界错位

新实现允许 capsule slots 仅在 sender 的前16个 block 读取长 source；后16层只能注意
4-slot 自身 causal submatrix。与 hard split 不同，hidden 和 autograd 路径完全连续，单测
确认 deep loss 会给 embeddings、早层和深层 writer 全部非零梯度。

零训练时 intervention 不改变 first16 的任何输出，但 parent all-layer F1 从 `.8188` 降到
`.7414`，bridge F1 从 `.8266` 降到 `.2826`。经过相同 train256/3200 updates 后，all-layer
恢复到 `.8633`，与 compute control `.8737` 的 delta `-.0104`，普通/cluster CI 下界均在
冻结 `-.08` safety margin 之上；correct-vs-shift first16 delta 更达到 `+.4909`。然而 first16
只有 `.7826`，低于 control `.8320`，delta `-.0495`，compact concentration 门失败。

关键不是 gradient 或因果性，而是 Transformer 状态与 cache 的一层错位：block 15 读源后
产生的 `h_16` 首次被投影为第17层 `K_16,V_16`，不在 layers 0--15 的 first16 packet 中。
因此唯一合理修正是 source-read-15：让 block 14 的输出 `h_15` 写入 packet 最后一层
`K_15,V_15`，再禁止后续读源。该 emission-aligned 版本必须在新的冻结集验证，不能回到
已打开的 confirm128 改 boundary。

### Emission-aligned source-read-15：新 96-case confirmation 四门全通过

排除全部历史 1,536 个 ID 后，eligible 只剩 103 条；事前冻结其中 96 条作为最后一组
MuSiQue confirmation（64 个 bridge clusters）。read-15 与 read-16、unrestricted control
使用完全相同 parent、train256、3,200 updates、loss、rank、INT4 和 seed。

read-15 first16 达到 `.781 EM / .831 F1`，而 read-16 是 `.646/.730`，unrestricted 是
`.677/.722`。paired F1：

- aligned vs unaligned `+.1010`，普通 CI `[+.0198,+.1833]`，cluster
  `[+.0081,+.1989]`；
- aligned vs unrestricted `+.1094`，普通 CI `[+.0313,+.1927]`，cluster
  `[+.0288,+.1975]`；
- aligned all vs unrestricted all `+.0292`，普通/cluster 下界 `-.0458/-.0534`，
  通过 `-.08` safety margin；
- correct vs shift first16 `+.6073`，普通 CI `[+.5146,+.6990]`，cluster
  `[+.5011,+.7116]`。

四道冻结门全部通过。read-15 训练 loss `.572` 也低于 read-16 `.721`；相邻 boundary
只改变 bottleneck state 是否落入实际发送的最后一个 K/V layer，因此这不是随意的
first-K 扫描。当前最强新主张是：**KV state compression 必须同时对齐计算图中的 hidden
bottleneck 与 wire format 中的 cache-layer boundary。** 下一步必须做零目标训练跨任务
复验和 serving 测量，不能再次使用已耗尽的 2Wiki-E 密封集。

### 2Wiki-200 零微调：alignment 的 in-domain 优势不迁移

按冻结协议在已开放的 2Wiki-200 上执行 aligned/unaligned first16/all。aligned read-15
first16 只有 `.241` F1，unaligned `.265`，unrestricted `.318`；aligned all `.236`，
unrestricted `.392`。paired gates：aligned-vs-unaligned first `-.0241`、CI 穿零；
aligned-vs-control first `-.0774`、CI `[-.1361,-.0192]`；all `-.1553`、CI
`[-.2118,-.0971]`；correct-vs-shift 虽为 `+.0468`，CI 下界 `-.0050`。四门全失败。

因此 emission alignment 解决了 **state 写到 wire packet 哪一层**，但没有解决 **写入
什么 task-invariant semantics**。MuSiQue stage/bridge loss 把 4-slot channel 压成了
relation-specific code。下一步固定 read-15 拓扑，不再改 boundary；以跨任务更强的
unrestricted pre-answer capsule 为 teacher 做行为蒸馏。该 teacher 与 student 同为
answer 前四槽 native KV，不包含此前 gold Tail teacher 的未来答案状态。

### Unrestricted pre-answer teacher：部分恢复跨任务，但五门仅一门通过

固定 read-15 student，用 unrestricted continued writer 作为冻结 teacher，在相同
stage/bridge teacher-forced 位置匹配 logits；两者都在 answer token 出现前发射4槽 native
KV。两步 loss-scale smoke 后固定 KL weight `.1`，未做权重扫描。

alignment-confirm96 上 first16 从 no-teacher `.831` 降到 `.771`，delta `-.0604`，普通和
cluster CI 下界约 `-.14`，in-domain safety 失败。2Wiki 上 first16 从 `.241` 回升到 `.271`，
但 delta `+.0297` 的 CI `[-.0224,+.0808]`；相对 unrestricted `.318` 仍低 `.0476`。
all 从 `.236` 回升到 `.290`，但仍比 unrestricted `.392` 低 `.1017`，CI 全负。只有
correct-vs-shift first16 `+.0682`、CI `[+.0146,+.1225]` 通过。

这表明 pre-answer teacher 方向不同于失败的 future-state teacher：它确实改善 EM、all
quality、generic first16 readout 和因果性；但全参数 student 仍在“in-domain 专化”和
“teacher 通用性”间折中，且 KL 出现少数重尾尖峰。下一版不调 KL，而应冻结 transferable
base，仅在 block14 后训练 zero-init boundary residual，使新增状态写入 `K15/V15`，同时
保持 layers0--14 与 teacher 完全一致。

## 不能过度声称的地方

1. **当前强结果适用于分别训练的同模型 sender/receiver。** Qwen 和 Llama 都已复现，
   但 Qwen Tail-KV 仍不能直接拼入 Llama；跨模型 sender/receiver 需要不会破坏开放词表
   identity 的结构化 adapter。
2. **这个 controlled task 的第一跳答案很短。** capsule 路径没有生成或传递答案
   明文，但一个允许 Agent A 先解码的系统仍可只传 3--6 个文本 token，其网络与
   receiver prefill 成本都低于 0.11--0.15 MiB KV。因此现象成立尚不等于端到端系统
   已优于 plaintext handoff。
3. **4-slot capsule 已出现可重复、跨数据集和跨模型族的因果信号，但没有普遍支配 post-answer Tail-KV。**
   Qwen Hotpot E290 上“不生成答案文本”仍有明显质量缺口；Llama E290 的 mean F1
   未分辨但 Tail EM 更高；2Wiki 上 first16 的 EM 更高、F1 未分辨；MuSiQue confirm256
   的 all-layer capsule 很强。它们共同支持 task-dependent Pareto frontier，而不是一种
   配置全面获胜。
4. **post-answer Tail-KV 受生成质量限制，pre-answer capsule 受状态抽取质量限制。**
   前者 source answer 不包含 gold 时显著下降；后者随数据、深度路由和 dataset 波动很大，
   两条路径都没有自动修复 Agent A 计算错误。
5. **重复事实需要 cluster-aware 复验。** 64 case 中有重复 entity pairs；论文结果
   必须扩大规模并按 bridge entity/pair 聚类 bootstrap。256-case 结果已按 108 个 bridge
   clusters 复验；新的 confirm256 也按 99 个 clusters 复验，但其它 benchmark 仍需同样处理。
6. **ID-disjoint 不等于 source-split-disjoint。** 新 256 cases 排除了全部 128 training
   IDs，但来自相同 MuSiQue train 原始 split；不能用它替代真正的跨 split/跨数据集结果。

## 下一步决策

下一阶段不应回到失败的池化 MLP，也不应只靠增加 slots。优先级如下：

1. **把深度当作 runtime state type，但保留 full-depth 联合训练。** last-half 并不通用；
   MuSiQue/2Wiki 都显示 first16 是 compact functional path，深层主要恢复 readout。朴素
   multi-budget、budget-conditioned loss 和 layer16 hard gradient boundary 已在两个独立
   数据集上失败。当前方法应固定为 all-layer end-to-end writer training，再在 serving 时
   选择 first16/all packet；不再训练彼此隔离的 base/residual。
2. **扩大通用 state-writing 数据，同时训练可压缩性。** N128→N512 显著提高 all-layer
   state，却恶化 post-hoc last16，说明只扩大数据会让表示扩散到更多深度。下一步用随机
   实体/关系、不同 dossier 长度和 query 模板扩展训练，并同时施加 layer-drop、null-state
   或多个 packet budgets；不继续单独加 rank。
3. **跨层重建而非局部 SVD。** 实际 INT4 已验证，下一压缩瓶颈是哪些深度必须发送。
   应学习 early functional base 加 selective deep lexical residual，或跨相邻层共享 basis；
   不再尝试已经失败的固定 head dropping、逐 token `8x128` SVD 或单一 hard last16 mask。
4. **token-identity-preserving cross-model adapter。** 将共享 lexical channel 与
   contextual residual 分开；先用大规模随机 text spans 做开放词表 teacher-response
   distillation，再在 MuSiQue 上学习 task residual。
5. **不显式输出短答案的计算任务。** 构造 Agent A state 位于长 reasoning、工具执行
   或多事实聚合之后，Agent B 需要复用分布/决策状态，而明文 handoff 需要长摘要或
   重新计算。只有这里才能证明相对 3-token 文本传输的系统价值。
6. **系统测量。** 报告 source prefill、4-slot emission、RoPE relocation、quant/dequant、
   network transfer、receiver TTFT，并与 plaintext summary、full KV、CacheBlend、
   KVPacket/RelayCaching 类方法使用相同链路和批量设置。

当前最值得继续的论文主线已经从“任意 KV 压成 8 个神奇 slots”收敛为：

> 在答案 token 出现前，用少量 learned state-emission slots 和 sender-only writer 从已有长上下文 cache 中
> 写出可重定位的 continuation state；下游任务可在不重放原文、也不接收答案明文的
> 情况下继续。post-answer Tail-KV 是强 teacher/upper bound，研究问题是怎样缩小
> pre-answer capsule 与 teacher 的质量差距，并做低比特、跨层和跨模型传输。

### 2026-08-18 冻结基座更新

后续实验改变了上述“不继续单独加 rank”的适用范围：那条结论针对会共同改写全部
capsule 参数的旧 writer。新的 frozen-base boundary residual 将 unrestricted 的 slots
与 layerwise writer 全部冻结，只在 block 14 后训练一个残差；因此它不是继续扩大旧
writer，而是在 15/16 个传输层严格不变的前提下控制 task-channel 容量。

rank-4 版本在 2Wiki first16 达到 `.338` F1，非劣于 unrestricted `.318`，并显著高于
full-parameter aligned `.241`；MuSiQue 从 base `.722` 提高到 `.763`，但置信区间跨零，
且低于 aligned `.831`。这证明了 base/task 分解能阻止跨任务遗忘，也把下一瓶颈收敛
为单层残差容量。允许一次预注册的 rank-16 follow-up；边界、slots、训练数据、步数、
学习率与 wire payload 均保持不变，不进行 rank sweep。

## 产物

- evaluator：`src/xmodel_kv/cli/evaluate_musique_tail_transplant.py`
- learned state-emission primitive：`src/xmodel_kv/soft_tail_capsule.py`
- capsule trainer/evaluator：
  `src/xmodel_kv/cli/train_musique_soft_tail_capsule.py`、
  `src/xmodel_kv/cli/evaluate_musique_soft_tail_capsule.py`
- paired statistics：`src/xmodel_kv/cli/analyze_paired_transfer.py`
- HotpotQA zero-shot evaluator/analyzer：
  `src/xmodel_kv/cli/evaluate_hotpot_summary_transfer.py`、
  `src/xmodel_kv/cli/analyze_hotpot_summary_transfer.py`
- HotpotQA frozen protocol/results：
  `docs/hotpot_zeroshot_capsule_preregister_20260817.md`、
  `docs/hotpot_zeroshot_capsule_results_20260817.md`
- HotpotQA pooled 200-case outputs：
  `outputs/amortized_semantic_kv/hotpot_zeroshot_writer_frozen200_pooled_20260817/`
- HotpotQA-E late-layer confirmatory protocol/results：
  `docs/hotpot_last18_confirm_preregister_20260817.md`、
  `docs/hotpot_last18_confirm_results_20260817.md`
- HotpotQA-E disjoint290 pooled outputs：
  `outputs/amortized_semantic_kv/hotpot_e290_last18_confirm_pooled_20260817/`
- Llama-3.1-8B no-search model-family replication protocol/results：
  `docs/llama31_writer_replication_preregister_20260817.md`、
  `docs/llama31_writer_replication_results_20260817.md`
- Llama-3.1-8B E290 pooled outputs：
  `outputs/amortized_semantic_kv/llama31_hotpot_e290_last16_replication_pooled_20260817/`
- Llama N512 scaling、packet-aware negative、depth localization 与 multi-budget negatives：
  `docs/llama31_writer_data_scaling_results_20260817.md`、
  `docs/llama31_packet_aware_writer_results_20260817.md`、
  `docs/llama31_half_depth_localization_diagnostic_20260817.md`、
  `docs/llama31_multibudget_writer_results_20260818.md`、
  `docs/llama31_progressive_base_residual_results_20260818.md`
- fresh MuSiQue confirm256 protocol/results：
  `docs/llama31_first16_confirm_preregister_20260817.md`、
  `docs/llama31_first16_confirm_results_20260817.md`
- 2WikiMQA zero-shot protocol/results/repeat：
  `docs/llama31_2wikimqa_first16_preregister_20260817.md`、
  `docs/llama31_2wikimqa_first16_results_20260817.md`、
  `outputs/amortized_semantic_kv/llama31_2wikimqa_first16_repeatability_20260817/`
- 2WikiMQA-E300 密封协议/结果/paired analysis：
  `docs/llama31_n768_2wikimqa_e_sealed_preregister_20260818.md`、
  `docs/llama31_n768_2wikimqa_e_sealed_results_20260818.md`、
  `outputs/amortized_semantic_kv/llama31_n768_2wikimqa_e_sealed_analysis_20260818/`
- actual INT4 codec：`src/xmodel_kv/capsule_codec.py`
- ICLR claim--evidence matrix：`docs/kv_capsule_iclr_claim_evidence_20260817.md`
- sealed soft-only capsule：
  `outputs/amortized_semantic_kv/musique_softcapsule_bridge2_s4_n128_sealedtest160_196_20260817/`
- rank-4 writer secondary correct/shift controls：
  `outputs/amortized_semantic_kv/musique_softcapsule_writer_r4_n128_secondary96_159_*_20260817/`
- rank-4 writer layer curve：
  `outputs/amortized_semantic_kv/musique_softcapsule_writer_r4_n128_secondary96_159_layers_*_20260817/`
- future-state distillation negatives：
  `outputs/amortized_semantic_kv/musique_softcapsule_distill_kl*20260817/`
- ID-disjoint 256-case frozen benchmark：
  `datasets/musique_handoff/test256_traincases_seed2031_exclude_train128.jsonl`
- 256-case writer correct/shift、paired cluster statistics 和 confidence-gate controls：
  `outputs/amortized_semantic_kv/preregistered_test256_writer_r4_s4_first30_int4_*_seed2031_20260817/`
- 256-case generated Tail-KV：
  `outputs/amortized_semantic_kv/preregistered_test256_tail_generated_int4*_seed2031_20260817/`
- RoPE transplant primitive：`src/xmodel_kv/policy_imprinting.py::stitch_history_cache`
- gold quant results：
  `outputs/amortized_semantic_kv/musique_tail_transplant_quant_qwen_gold_dev32_95_20260817/`
- generated quant results：
  `outputs/amortized_semantic_kv/musique_tail_transplant_quant_qwen_generated_dev32_95_20260817/`
- gold first-K results：
  `outputs/amortized_semantic_kv/musique_tail_transplant_firstk_int4_qwen_gold_dev32_95_20260817/`
- generated first-K results：
  `outputs/amortized_semantic_kv/musique_tail_transplant_firstk_int4_qwen_generated_dev32_95_20260817/`
- head ablation（gold/generated）：
  `outputs/amortized_semantic_kv/musique_tail_transplant_heads_first21_int4_qwen_*_dev32_95_20260817/`
- head-rank curve：
  `outputs/amortized_semantic_kv/musique_tail_transplant_headrank_first21_qwen_gold_dev32_95_20260817/`
- learned-codec 结果：`outputs/amortized_semantic_kv/musique_tailkv_*n128_dev32_95*20260817/`

复现实验（generated；gold 只需改 `--source-state` 和输出目录）：

```bash
.venv/bin/python -m xmodel_kv.cli.evaluate_musique_tail_transplant \
  --dataset datasets/musique_handoff/dev197_seed2027.jsonl \
  --model Qwen/Qwen3-8B \
  --output-dir outputs/amortized_semantic_kv/musique_tail_transplant_quant_qwen_generated_dev32_95_20260817 \
  --device cuda:0 --source-state generated_answer \
  --eval-offset 32 --eval-count 64 --max-new-tokens 24 \
  --quant-bits 16,8,4
```
