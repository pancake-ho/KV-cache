# MuSiQue 3/4-Agent KV Reuse Pilot

## Scope

This pilot maps the real MuSiQue hop count to the actual number of inference
agents: 3-hop is A -> B -> C and 4-hop is A -> B -> C -> D.  Twenty frozen dev
examples are used for each setting (seed 2027).  Every downstream agent performs
a new relation lookup over eight candidates, so per-stage chance accuracy is
12.5%.  The correct chain, questions, answers, and supporting evidence come from
MuSiQue-Answerable v1.0.  Train rows supply only real distractor records; no
parameters or thresholds are trained.

This is a controlled semi-synthetic evaluation because each downstream evidence
set is rendered as an explicit `LOOKUP_KEY`/`RESULT` table.  Agent A still reads a
long dossier: its Llama-3.1-8B prompt is 5,507--6,136 tokens for the 3-agent set
and 5,683--6,130 tokens for the 4-agent set.  Thus all tested shared histories are
longer than 4K tokens.

## Methods

- **Summary-chain:** each receiving agent starts a fresh call with only the
  immediately preceding agent's text and its own task.
- **Native-front:** the receiving agent's system prompt is first and its complete
  cumulative plaintext history is densely prefetched from scratch.
- **Native-tail:** start from A's serialized conversation, append each new agent's
  policy/task at the tail, and densely prefill the whole growing sequence.
- **Exact Tail-KV:** use exactly the same serialized sequence as Native-tail but
  continue from its live exact KV cache.  This is not KVComm.
- **KVComm-dense:** the local released KVComm segmented template, with all target
  calls forced through its dense path.  This isolates the template from offsets.
- **KVComm:** the same template and online sample order, with the released
  Anchor/Offset scheduler allowed to choose dense or reuse.

## Accuracy

All entries are strict exact match over 20 examples.  "All-chain" requires every
downstream handoff in the sample to be correct.

| Agents | Method | Final | All-chain | Per downstream stage |
|---:|---|---:|---:|---:|
| 3 | Summary-chain | 15/20 (75%) | 15/20 (75%) | 20/20, 15/20 |
| 3 | Native-front | 16/20 (80%) | 16/20 (80%) | 20/20, 16/20 |
| 3 | Native-tail | 18/20 (90%) | 17/20 (85%) | 19/20, 18/20 |
| 3 | Exact Tail-KV | 18/20 (90%) | 17/20 (85%) | 19/20, 18/20 |
| 3 | KVComm-dense | 17/20 (85%) | 16/20 (80%) | 19/20, 17/20 |
| 3 | KVComm Anchor/Offset | 16/20 (80%) | 15/20 (75%) | 18/20, 16/20 |
| 4 | Summary-chain | 9/20 (45%) | 5/20 (25%) | 18/20, 10/20, 9/20 |
| 4 | Native-front | 17/20 (85%) | 11/20 (55%) | 19/20, 13/20, 17/20 |
| 4 | Native-tail | 16/20 (80%) | 8/20 (40%) | 19/20, 10/20, 16/20 |
| 4 | Exact Tail-KV | 16/20 (80%) | 8/20 (40%) | 19/20, 10/20, 16/20 |
| 4 | KVComm-dense | 11/20 (55%) | 9/20 (45%) | 18/20, 14/20, 11/20 |
| 4 | KVComm Anchor/Offset | 10/20 (50%) | 6/20 (30%) | 12/20, 10/20, 10/20 |

Exact Tail-KV and Native-tail have 100% step-level generated-text agreement in
both graphs.  Their quality is therefore identical by construction; the cache
path itself introduces no approximation.

## Prefill and KVComm activation

The Tail comparison counts downstream prefill tokens and excludes the common A
call.  The KVComm comparison counts per-sample placeholder materialization plus
target model prefill and excludes its one-time static-template setup.  Absolute
token totals between these two families should not be compared directly; the
paired reductions within each family are meaningful.

| Agents | Paired comparison | Dense tokens | Reuse tokens | Reduction | Other signal |
|---:|---|---:|---:|---:|---:|
| 3 | Native-tail -> Exact Tail-KV | 16,091.95 | 2,977.45 | 81.50% | 100% text agreement |
| 4 | Native-tail -> Exact Tail-KV | 26,158.60 | 4,303.05 | 83.55% | 100% text agreement |
| 3 | KVComm-dense -> KVComm | 28,762.40 | 22,614.85 | 21.37% | 19/60 calls reused; mean TTFT -26.29% |
| 4 | KVComm-dense -> KVComm | 37,334.90 | 24,337.25 | 34.81% | 39/80 calls reused; mean TTFT -41.85% |

For 3 agents, paired KVComm changes one dense-correct final sample to wrong and
changes no dense-wrong sample to correct.  For 4 agents, three dense-correct
final samples become wrong while two dense-wrong samples become correct.  At the
all-chain level the 4-agent transition is four correct-to-wrong versus one
wrong-to-correct.  This is end-to-end causal evidence for the enabled method,
although later calls also inherit any earlier KVComm output error.

## Interpretation and limitations

The earlier observation that more agents improved GSM8K is not a general
multi-agent effect.  On an actual sequential dependency chain, additional
handoffs amplify errors: summary-only drops from 75% final accuracy at three
agents to 45% at four agents.  Exact tail continuation remains cheap and exact,
but it works here because the receiver policy is explicitly reasserted at the
tail; this pilot does not solve the trusted-front policy-imprinting problem.

The full Native-front-to-KVComm gap must not be attributed entirely to offsets.
KVComm's released graph template injects the original request and direct
predecessor output, whereas Native-front retains the complete cumulative
plaintext transcript.  The fair offset ablation is KVComm-dense versus KVComm:
85% -> 80% final accuracy for three agents and 55% -> 50% for four agents, with
the larger 45% -> 30% all-chain loss in the four-agent case.  The current result
is a 20-example controlled pilot; a paper table still needs more seeds/examples,
semantic scoring for long MuSiQue answers, and a natural-text candidate variant.
