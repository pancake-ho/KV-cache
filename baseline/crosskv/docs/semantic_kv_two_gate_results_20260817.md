# Semantic-KV 两道门验证（2026-08-17）

## 结论

本轮把“摘要 KV”拆成了两个必须依次回答的问题：

1. **Gate 1：短的、接收模型原生的 KV 状态是否存在？**
2. **Gate 2：能否用一个跨样本、跨模型的映射器，从 Agent A 已有的 KV 中摊销地构造该状态？**

实验结论是：

- **Gate 1 明确通过。** 在不使用任何 CoQA 问答进行构造的条件下，Llama
  的 8-slot receiver-native KV 在 8 个文档、64 个未见问题上全部优于无摘要基线。
- **Gate 2 只在短文档同分布上得到弱正信号，尚未通过目标场景。** Qwen3-8B
  到 Llama-3.1-8B 的共享映射器在未见 CoQA 文档上优于无摘要，但只保留约
  16%--22% 的增量质量。
- **多文档依赖传递失败。** 把同一个映射器直接用于平均 6514-token 的 MuSiQue
  两阶段任务时，8-slot 和 16-slot student 都只有 1/8，等于无摘要的随机水平；
  显式文本 handoff teacher 为 8/8。
- **当前瓶颈不是 INT4，也不是“短状态不存在”。** 它是从长源缓存中做有覆盖、
  有任务条件的语义读出，以及把这种读出泛化到不同文档结构和依赖关系。

因此目前支持的最强表述是：

> 短的 receiver-native semantic KV state 存在；一个简单的、仅用最终层 tail
> readout 训练的小规模跨模型 transcoder 可以在同分布上学到部分信号，但尚不能
> 从长、多文档、依赖推理缓存中可靠提取可复用状态。

这还不是可投稿的完整方法，但两道门已经把存在性、构造、量化和任务泛化四个因素
分开，明确了下一步真正需要解决的位置。

## 实验设置

### Gate 1：receiver-native existence oracle

- 接收模型：`meta-llama/Llama-3.1-8B-Instruct`。
- 数据：CoQA validation 的 conversation 0--7；每个文档 8 个未见问题。
- 构造目标：只使用四个固定的 self-study prompt（事实概述、实体关系、事件顺序、
  具体属性），不使用任何 CoQA 问题、答案或其 teacher logits。
- 优化对象：每个文档独立的 8/16 个 Llama-native KV slots；模型权重冻结。
- 初始化：对完整 story cache 做 source-pool；每个预算优化 300 步。
- 该门刻意消除了 source-model identity，只检验短接收状态是否存在，不代表已完成
  Qwen-to-Llama 映射。

### Gate 2：amortized cross-model transcoder

- 源模型：`Qwen/Qwen3-8B`。
- 接收模型：`meta-llama/Llama-3.1-8B-Instruct`。
- 两个模型均冻结；只训练一个所有文档共享的 `TailReadoutKVTranscoder`。
- transcoder 在完整 Qwen cache 末尾加入可学习 readout slots，从最终 source hidden
  states 得到 256 维/slot latent，再按 Llama 的 layer/head 结构解码成 receiver-native
  K/V，并在接收方紧凑位置重新施加 RoPE。
- 训练只使用四类 generic self-study responses 的 CE/KL，以及 receiver target-KV
  辅助损失；不使用 CoQA QA 标签。
- CoQA train document 0--15，test document 16--23，严格文档不重叠；1200 步。
- 分别训练 8-slot 与 16-slot 模型；测试 FP16、模拟 INT8 和模拟 INT4 latent。

### 依赖文档压力测试

- 使用 MuSiQue 构造两阶段 handoff：Agent A 在长、多文档 dossier 中确定中间答案，
  Agent B 根据该答案在 `LOOKUP_KEY -> RESULT` 表中做精确查找。
- dense teacher 显式获得正确中间答案，用来隔离 Stage-B 能力；student 只获得压缩
  KV；no-summary 不获得 Agent A 信息。
- 使用 CoQA 上训练好的 transcoder，**不在 MuSiQue 上重训**，测试的是任务、长度和
  文档结构的联合外推。

## Gate 1 结果：通过

以下是 conversation 0--7 的宏平均；每个 conversation 等权。置信区间是每文档
相对 no-summary 增益的探索性 paired-t 95% CI。

| 接收 KV | 初始 F1 | 优化后 F1 | no-summary F1 | self-study text F1 | full-story teacher F1 | 相对 no-summary 增益（95% CI） | 胜出文档 | 源段 slot 压缩 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 slots | 0.104 | **0.357** | 0.074 | 0.599 | 0.934 | **+0.282 [0.118, 0.447]** | **8/8** | 47.5x |
| 16 slots | 0.113 | **0.346** | 0.074 | 0.599 | 0.934 | **+0.271 [0.126, 0.416]** | **8/8** | 23.8x |

- 8 slots 保留了 full-story teacher 增量 F1 的 32.9%，以及同一 self-study 内容以
  明文传递时增量 F1 的 53.8%。
- 16 slots 对应为 31.6% 和 51.7%。它的生成 F1 没有超过 8 slots，但 teacher KL
  更低（1.795 vs 2.063），说明非单调主要仍是优化问题，不能据此声称 8 slots
  容量更大。
- source-pool 初始状态接近无摘要，提升来自优化后的 receiver-native state，不只是
  pooling 偶然保留了答案。

判定：**Gate 1 PASS**。在接收方模型里，短、query-agnostic、可供未见问题复用的
KV 状态确实存在。

原始结果：

- `outputs/semantic_kv_summary/gate1_llama31_8b_selfstudy_b8_b16_s300_idx0003_20260817/`
- `outputs/semantic_kv_summary/gate1_llama31_8b_selfstudy_b8_b16_s300_idx0407_20260817/`

## Gate 2A 结果：同分布弱正信号

CoQA test 只包含训练时未见的 document 16--23；表中比较均使用各预算的 FP16
评估，避免把独立重复生成产生的小数值波动当成量化收益。

| 映射器 | student F1 | no-summary F1 | teacher F1 | 增益 | 增量质量保留率 | 胜出文档 | 探索性 95% CI | receiver BF16 KV | INT4 latent payload |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 slots | **0.157** | 0.044 | 0.559 | +0.113 | 22.0% | 6/8 | [-0.024, 0.251] | 1 MiB | 1024 B |
| 16 slots | **0.126** | 0.044 | 0.559 | +0.083 | 16.1% | 6/8（另 1 tie） | [0.023, 0.143] | 2 MiB | 2048 B |

量化结果：

| 映射器 | FP16 F1 | INT8 F1 | INT4 F1 |
|---:|---:|---:|---:|
| 8 slots | 0.1570 | 0.1566 | 0.1570 |
| 16 slots | 0.1265 | 0.1267 | 0.1472 |

INT4 下 16-slot 的表观提升不应解释为真实量化收益，因为每次量化评估都重新生成，
teacher 也有轻微数值波动。这里能支持的结论仅是：**latent 从 FP16 压到模拟 INT4
没有导致信号崩溃**。

8-slot held-out story 平均 335.6 个 Qwen tokens。Qwen3-8B 的 BF16 full KV 约为
47.2 MiB/文档；转成 1 MiB 的 Llama receiver KV 是约 47.2x 的接收侧 HBM 缩减。
若网络只传输 8x256 的 INT4 latent，则原始位 payload 是 1024 B（加每 slot scale
约 32 B），相对源 full KV 约 4.7 万倍缩小。16-slot 对应约 23.6x HBM 缩减和约
2.3 万倍 wire 缩减。这里的 wire ratio 很大，但质量只保留 16%--22%，不能单独
作为系统贡献。

判定：**Gate 2A WEAK POSITIVE**。共享跨模型映射器不是完全学不到，但样本只有
16 train / 8 test documents，效果低且不稳定，不足以宣称 Gate 2 通过。

原始结果与 checkpoints：

- `outputs/amortized_semantic_kv/gate2_qwen8b_to_llama8b_b8_z256_train16_test8_s1200_20260817/`
- `outputs/amortized_semantic_kv/gate2_qwen8b_to_llama8b_b16_z256_train16_test8_s1200_20260817/`

## Gate 2B 结果：多文档依赖传递失败

| 映射器 | 平均 source tokens | teacher accuracy | student accuracy | no-summary accuracy | student 胜出数 | INT4 payload |
|---:|---:|---:|---:|---:|---:|---:|
| 8 slots | 6514.25 | **8/8** | **1/8** | **1/8** | 0/8 | 1024 B |
| 16 slots | 6514.25 | **8/8** | **1/8** | **1/8** | 0/8 | 2048 B |

两个 student 命中的都是同一个默认/首候选式 case，整体等于 8-way lookup 的随机
水平。因为 controlled teacher 为 100%，失败不能归因于 Llama 不会做 Stage-B lookup；
因为 8/16-slot 的 INT4 CoQA 信号均未崩溃，也不能优先归因于量化。最直接的解释是：

1. 当前 readout 只读取最终层的一小段 tail hidden states，未覆盖 6.5k-token
   多文档中的稀疏关键事实；
2. generic self-study 目标是 query-agnostic 的文档概括，没有用 Stage-B 所需关系来
   条件化抽取；
3. 16 个短 CoQA train documents 无法支撑长度、结构和任务的联合外推；
4. 单一的稠密 semantic latent 容易保留主题和概述，却丢失 entity/value 精确绑定。

判定：**完整 Gate 2 FAIL**。当前 transcoder 还不能完成目标中的“非独立文档 +
摘要 KV + 跨模型 + 压缩传输”。

有效的 controlled-v2 结果：

- `outputs/amortized_semantic_kv/musique_handoff_controlled_v2_b8_int4_n8_20260817/`
- `outputs/amortized_semantic_kv/musique_handoff_controlled_v2_b16_int4_n8_20260817/`

此前没有把 teacher lookup 行为控制好的 `musique_handoff_*` 和
`musique_handoff_controlled_*` 目录只属于 prompt-design diagnostics，不应进入论文
结果。

## 下一步实验决策

下一轮不再单独增加 slot 数，而做一个最小但能针对失败机理的版本：

1. **关系/接收任务条件化 readout。** 将 Agent B 的任务描述或 relation query 作为
   readout query，让 source KV 中被抽取的内容随下游需求变化；这仍不需要传明文
   document，也不同于预知最终答案。
2. **两类 slot。** 使用例如 8 个 global semantic slots 加 8 个 sparse evidence
   slots。前者保留概述，后者通过跨层 token selection 或 attention pooling 保留精确
   entity/value binding。
3. **多层源特征。** 至少比较 final-layer-only、early/mid/late 三层融合和直接
   K/V-content readout，验证失败是否来自 final hidden state 的覆盖瓶颈。
4. **在依赖任务分布上训练。** 用数百到数千个 MuSiQue/HotpotQA train case 训练，
   按 document、relation chain 和答案实体严格切分；保留当前 controlled Stage-B
   lookup 作为第一阶段单元测试，再回到开放式 QA。
5. **固定三条判定线。** 同时报告 no-summary、明文 handoff、full-context；只有
   semantic KV 在未见依赖链上稳定胜过 no-summary，且保留明文 handoff 的显著增量，
   才进入端到端 TTFT/传输 benchmark。

这个顺序能直接检验论文核心：优势应来自“把已有 KV 变成可摘要、可跨模型、可低比特
传输的 agent state”，而不是只得到一个极端压缩但语义失效的数据包。

## 实现与验证

本轮新增：

- `src/xmodel_kv/semantic_kv_transcoder.py`：tail-readout 跨模型 transcoder、target
  RoPE、INT8/INT4 fake quantization。
- `src/xmodel_kv/cli/probe_amortized_semantic_kv.py`：CoQA 摊销训练和 held-out eval。
- `src/xmodel_kv/cli/evaluate_amortized_musique_handoff.py`：受控 MuSiQue handoff。
- `tests/test_semantic_kv_transcoder.py`：shape、cache、冻结模型反向传播、量化 STE。
- 修复 Llama 空 tools chat-template 被误渲染为 function call，以及 target stop-token
  进入答案评分的问题。

完整测试：`105 passed, 2 warnings`；warning 来自已有 policy capsule Transformer，
与本轮新增路径无关。

## 复现命令

Gate 1（另一个进程将 indices 改为 `4,5,6,7`）：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m xmodel_kv.cli.probe_semantic_kv_summary \
  --dataset datasets/coqa/data/validation-00000-of-00001.parquet \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir outputs/semantic_kv_summary/gate1_llama31_8b_selfstudy_b8_b16_s300_idx0003_20260817 \
  --device cuda:0 --conversation-indices 0,1,2,3 --budgets 8,16 \
  --max-questions 8 --training-mode self_study --initialization source_pool \
  --self-study-max-new-tokens 96 --steps 300 --learning-rate 0.01 \
  --log-every 50 --max-new-tokens 24
```

Gate 2（8 slots；16-slot run 只需改 `--slots` 和输出目录）：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m xmodel_kv.cli.probe_amortized_semantic_kv \
  --dataset datasets/coqa/data/validation-00000-of-00001.parquet \
  --source-model Qwen/Qwen3-8B \
  --target-model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir outputs/amortized_semantic_kv/gate2_qwen8b_to_llama8b_b8_z256_train16_test8_s1200_20260817 \
  --device cuda:0 --train-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --test-indices 16,17,18,19,20,21,22,23 --slots 8 --latent-dim 256 \
  --max-questions 8 --self-study-max-new-tokens 64 --steps 1200 \
  --learning-rate 0.001 --log-every 100 --max-new-tokens 24
```

MuSiQue controlled handoff：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m xmodel_kv.cli.evaluate_amortized_musique_handoff \
  --dataset datasets/musique_handoff/dev197_seed2027.jsonl \
  --checkpoint outputs/amortized_semantic_kv/gate2_qwen8b_to_llama8b_b8_z256_train16_test8_s1200_20260817/transcoder.pt \
  --source-model Qwen/Qwen3-8B \
  --target-model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir outputs/amortized_semantic_kv/musique_handoff_controlled_v2_b8_int4_n8_20260817 \
  --device cuda:0 --indices 0,1,2,3,4,5,6,7 --quant-bits 4 \
  --max-new-tokens 24
```
