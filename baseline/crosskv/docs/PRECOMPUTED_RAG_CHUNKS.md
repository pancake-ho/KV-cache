# Precomputed RAG chunk KV prototype

This prototype separates document-KV production from question answering. The
consumer receives no document token IDs and performs no document prefill.

## Producer and consumer semantics

For a fixed system prompt `S`, the independent producer evaluates each
document as `S + Di` and stores only the `Di` portion of every layer's K/V on
CPU. System KV is stored once. Therefore every document is conditioned on the
same system prompt but not on other documents.

At consumption time, selected chunks are concatenated in a requested order.
Keys are moved from their producer RoPE coordinates to their target virtual
coordinates by inverse/source and forward/target rotations. Values are not
rotated. The consumer processes only question and output tokens, one token at
a time, while document K/V remains in CPU micro-pages.

The order-specific control evaluates `S + D1 + ... + Dn` once and slices the
result into chunks. It retains ordinary cross-document causal conditioning and
tests the producer/consumer plumbing, but its chunks are not independently
reorderable.

## Page policies

All policies retain complete native-precision document KV in CPU memory. Pages
contain 16 tokens and never cross document boundaries.

- `independent_full` loads every page.
- `independent_sparse` selects about 5% of pages independently at each layer
  and token using a sign-aware key min/max bound.
- `independent_seeded` uses the same 5% attention budget. Chunk title pages and
  pages with high question-token IDF overlap consume 75% of the budget; the
  remaining pages are selected dynamically from the KV directory.

The seed is a page index, not an irreversible token-pruning operation. Any cold
page can still be fetched later.

## Initial Qwen2.5-7B result

Eight HotpotQA-style examples contained a mean of 4,491 document tokens and
nine document chunks. The independent path moved a mean of 3,595 document
tokens to new RoPE positions. Consumer document prefill was zero in every run.

| Mode | EM | F1 | Token fraction attended | Cumulative bytes / full KV |
| --- | ---: | ---: | ---: | ---: |
| Ordinary full prompt | 0.500 | 0.646 | 100% | n/a |
| Order-specific full chunk KV | 0.500 | 0.646 | 100% | 100% |
| Independent full chunk KV | 0.250 | 0.458 | 100% | 100% |
| Independent dynamic pages | 0.000 | 0.083 | 5.18% | 59.6% |
| Independent seeded pages | 0.375 | 0.438 | 5.19% | 15.8% |

The full cold KV volume was 2,060,369,920 bytes across the eight examples.
Purely dynamic paging transferred 1,227,495,424 bytes because its small cache
thrashed across question and decode tokens. Seeding reduced this to 325,373,952
bytes and raised the mean page-hit fraction to 92.4%.

These eight examples are a functionality screen, not a paper-quality accuracy
result. The Python/Hugging Face consumer is slower than ordinary prefill and is
not a latency implementation. The order-specific control matches aggregate
task quality but is not bit-identical to batched full prefill because question
tokens are processed through a different streaming numerical path.

## Remaining system boundary

The current bundle is CPU-resident in one Python process. It has not yet been
serialized into LMCache for a separate process, and vLLM still has no scheduler
contract for allocating only the sparse scratch working set. Variable system
prompts also require either a cache key containing the system hash or a
reconditioning mechanism. The independent-full quality gap measures the cost
of removing cross-document causal conditioning and must be studied on a larger
set before optimizing the runtime.

Raw results are in
`outputs/precomputed_rag_chunks/qwen25_7b_hotpot8_p16_r005_c2_seed075_eosfix`.
