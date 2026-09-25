# Agent policy imprinting 预实验协议

## 1. 要证伪的命题

固定同一个 decoder-only 模型 $M$，仅改变 source/target agent 的私有前缀：

\[
Z_s=P_s\oplus \mathcal T_s,\qquad Z_t=P_t\oplus \mathcal T_t.
\]

共享历史 $H$ 的文本和 token ID 逐个相同。待检验的命题是：在 source agent 下形成的
历史 KV 已经吸收了 $Z_s$ 的 policy/tool-schema 信息；即使删除 source prefix、把 H
重新定位到原生 target prefix 后面，target 的下一动作仍会受到 source policy 的定向影响。

零假设 $H_0$：完成正确的 cache slicing、RoPE 位置校正和 target-private prefix 拼接后，
hybrid cache 与 target-native cache 的动作差异不超过 identity cache-surgery 误差。

备择假设 $H_1$：hybrid cache 相对 target-native cache 出现可重复的任务损失，而且动作概率
或生成结果系统性朝 source-native action 偏移。

## 2. 四个实验臂

令 $G$ 为 target 原生处理的 assistant generation header。KV cache 本身不能产生最后一个
token 的 residual/logits，所以所有臂都必须经过同一个 $G$。

1. **A / Target Native reference**：原生 prefill $Z_t\oplus H$，再处理 $G$。
2. **S / Source Native reference**：原生 prefill $Z_s\oplus H$，再处理 $G$。
3. **I / Identity Stitch**：从 target-native cache 中切出 $H$，再接回 target prefix，处理 $G$。
4. **B / Policy Swap**：从 source-native cache 中切出 $H$，去除 source-position RoPE、施加
   target-position RoPE，接到原生 $Z_t$ 后，再处理 $G$。

后缀修复臂 $B_r$ 只转移 $H[:-r]$，由 target 原生重算最后 $r$ 个历史 tokens，再处理
$G$。预注册 $r\in\{0,8,32,128,512,\mathrm{full}\}$。

## 3. 必须固定的因果控制

- 固定 $M_s=M_t$，第一阶段不使用 ridge、MLP 或任何跨模型 mapper。
- source/target 的 $H$ 和 $G$ 必须是完全相同的 token ID，不能只比较字符串。
- clean subset 使用等 token 长度的镜像 prompt/schema；真实不等长 prefix 单独报告。
- $H$ 使用外生、固定的 user/tool/neutral assistant 文本，不能包含泄露 source 角色的生成内容。
- source/target 方向成对互换，排除工具名、颜色、首尾位置等先验偏差。
- I 必须与 A 保持相同 tool call；identity action KL 应小于 $10^{-5}$。
- 去 RoPE 后，第 0 层 H-KV 应接近相同；若第 0 层就有明显差异，应先检查 token boundary、
  cache slice、position ID 和 attention mask。

## 4. 实验因子

第一轮受控筛查包含四类任务：

| family | 仅改变什么 | target action 依赖什么 | 作用 |
|---|---|---|---|
| tool name | system policy | 指定工具名 | 简单 policy-only 边界 |
| tool argument | system policy | 指定 JSON 字段 | 参数级 policy-only 边界 |
| tool schema | tool descriptions | schema 中的 mandatory/forbidden | schema-only 因果测试 |
| history selection | system policy | 从相同 H 选择 first/last record | policy-conditioned 历史解释 |

每类做 source/target 双方向，并扫共享历史长度约
$20/300/1{,}100/4{,}500$ tokens。正式统计实验应扩展到
$128/1\mathrm K/4\mathrm K/16\mathrm K$ tokens，每个 family/length/direction 至少 25 个
预先生成的不同实例，并随机化 record value、tool 顺序、相关证据位置和等 token 长度的
policy 别名。不能把同一个 prompt 的长度扫描当作独立样本。

若受控实验为正，再在 BFCL/ToolSandbox/τ²-Bench agent traces 上做同样的成对构造。合成任务
只能证明机制存在，不能单独证明真实部署中的发生率。

## 5. 预注册指标

### Primary endpoint

- target ground-truth full tool-call EM（tool name + canonicalized JSON arguments）。
- target-native agreement。
- 配对差值：$\mathrm{EM}(B)-\mathrm{EM}(A)$，同时报告全量集合和
  A/S 均正确且动作互斥的 eligible subset。

### Directional source leakage

令 $y_s,y_t$ 分别为 source/target native action：

\[
D_i=
\left[\log p_B(y_s)-\log p_B(y_t)\right]
-
\left[\log p_A(y_s)-\log p_A(y_t)\right].
\]

$D_i>0$ 表示 hybrid 状态向 source action 定向偏移。只报告 B 变差而没有该方向性证据，
不能排除随机 cache corruption。

### Distribution and mechanism diagnostics

- 在 target-native action 路径上 teacher-force，报告 full-vocabulary sequence KL、最大单 token
  KL、NLL delta 和 top-1 agreement。
- identity-adjusted counterfactual gap：

\[
G_i=D_{KL}(A\|B)-D_{KL}(A\|I).
\]

- 对每层 H-K/V 去 RoPE 后报告 relative-L2 与 cosine；后续增加 layer-block patching，验证
  哪些层的 source H-KV 会因果改变动作。
- 二元 EM 使用 paired bootstrap 95% CI 和 McNemar test；连续 leakage/KL 使用 paired
  bootstrap CI。No-Go 结论应做等价性检验，而不是把“不显著”直接当作“无效应”。

## 6. Go / No-Go 标准

进入论文主线至少需要：

1. identity tool-call agreement 为 100%，mean action KL $<10^{-5}$；
2. eligible rate 至少 90%，证明任务本身确实受 source/target policy 控制；
3. 在至少两个非同义任务 family 和一个自然 agent benchmark 上，B 相对 A 的 full-call EM
   下降至少 5 pp，paired 95% CI 下界大于 0；
4. source leakage $D_i$ 的中位数为正且 CI 不跨 0，并出现直接生成 source action 的案例；
5. 效应在双方向和至少两个模型规模上复现；
6. full target replay 恢复 A，而短 replay 曲线暴露可利用的 correction budget。

若在强 policy/schema 冲突、16K 历史、两个模型和自然任务上，EM 差值上界仍小于 1--2 pp，
且没有方向性 leakage，则否定“counterfactual history state 是主要问题”这条主线。若只在
history-conditioned selection/interpretation 上失败，应收窄命题，不能写成所有 agent switch
都会失败。

## 7. 当前 Qwen3-8B pilot（不作为最终统计结果）

当前实现使用等长 prefix，32 个受控零-replay 格子全部满足 source/target native action
互斥，identity agreement 为 32/32，最大 identity mean-KL 为 $6.2\times10^{-7}$。

- hybrid full-call 失败 6/32；其中 history-selection 失败 5/8，tool-schema 失败 1/8；
  简单 tool-name 和固定 argument 各为 0/8。
- 30/32 个格子的 source-action margin 朝 source 方向移动。
- 一个 5,140-token schema-only case 中，target-native 调 `route_red`，source-native 调
  `route_blue`，hybrid 直接错误调用 `route_blue`；action KL 为 0.281。
- history-selection 的长历史 case 中，hybrid 多次直接提交 source 选择的 record；全量 target
  replay 恢复 target action，而 512-token replay 在长历史上仍可能失败。
- layer 0 去 RoPE 后 H-K/V 距离为 0；差异从后续层出现，并在中后层明显增大。

这些结果已经证明代码路径中存在行为级 positive case，但样本是合成且高度受控的，当前只能
支持“现象值得继续”，还不能支持论文中的普遍性结论。后续 BFCL 自然历史、Qwen3-14B、
K/V-layer-token causal patching 和 replay curve 的结果已完成，见
[`policy_imprinting_findings.md`](policy_imprinting_findings.md)。

## 8. 运行方式

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m xmodel_kv.cli.policy_imprinting \
  --model Qwen/Qwen3-8B \
  --device cuda:0 \
  --scenarios tool_name,tool_argument,tool_schema,history_selection \
  --directions forward,reverse \
  --history-turn-pairs 0,8,32,128 \
  --replay-tokens 0,8,32,128,512,100000 \
  --output outputs/policy_imprinting/qwen3_8b_preexperiment.jsonl
```

`100000` 会被裁剪为完整 H 长度，等价于 full target replay。每行包含原生 A/S、identity I、
所有 $B_r$、tool/argument EM、action KL、source leakage 和逐层 H-K/V 距离。

决定性三向对照使用 9-way 动作空间、matched/null/third 三个 source arm，并将等长 clean 与
re-RoPE shifted 分开运行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run xkv-policy-triad \
  --model Qwen/Qwen3-8B \
  --device cuda:0 \
  --families position,content \
  --arms null,matched,third \
  --layouts clean,shifted \
  --replicates 25 \
  --record-count 9 \
  --seed 2027 \
  --replay-tokens 0,100000 \
  --output outputs/policy_imprinting/qwen3_8b_policy_triad_n25_seed2027.jsonl
```

多 GPU 时用 `--num-shards N --shard-index i` 按完整 matched/null/third group 分片，最后用
`xkv-summarize-policy-triad ... --expected-rows 600` 检查缺失、重复和每个 cell 的 Wilson 95%
置信区间。正式报告中每个方向的独立 history 数为 n=25；arm/layout 重复测量不能叠加成更大 n。
