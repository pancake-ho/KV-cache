# Demand-paged reusable KV pilot

This pilot isolates decode-time document-KV paging from CacheBlend's
cross-context correction. It uses exact BF16/FP16 KV produced by a normal
prefill, places the document range in CPU memory, keeps only a small page
directory and the non-document prompt state on GPU, and selects document pages
from the current query independently at every decoder layer.

## Why this is not implemented inside CacheBlend

LMCache CacheBlend retrieves cached chunks layer by layer during prefill,
repairs selected tokens, and finally writes the complete combined KV back to
vLLM's paged GPU cache. Demand paging changes which KV is visible during every
decode attention call, so the eventual serving implementation needs a vLLM
attention backend in addition to CacheBlend. LMCache remains the intended
document store and context-repair front end.

The pilot deliberately does not include quantization, learned transforms,
cross-position reuse, or real SSD I/O. Those factors should only be added after
the query-aware paging path has been measured on exact, current-request KV.

## Selector

Cold document KV is split into physical pages containing all KV heads for one
layer. For every page and KV head, the GPU-resident directory stores the
component-wise minimum and maximum key. Given one decoder query, the selector
computes a sign-aware upper bound on every query-key logit and loads the top
pages. GQA query heads are reduced by maximum score because a physical page is
loaded once for the shared KV heads.

Selected pages remain in a bounded per-layer working set. Results report both
logical selected-token fraction and physical CPU-to-GPU page misses.

## Run

```bash
cd /path/to/CrossKV
./.venv/bin/python -m xmodel_kv.cli.probe_demand_paged_kv \
  --dataset datasets/hotpot_summary_transfer/hotpotqa_e_disjoint_from_standard.jsonl \
  --model ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<revision> \
  --output-dir outputs/demand_paged_kv/qwen25_7b_hotpot_p64_r005 \
  --device cuda:0 \
  --count 8 \
  --page-size 64 \
  --budget-ratio 0.05 \
  --cache-capacity-ratio 2
```

The command writes paired full-KV and demand-paged generations, EM/F1, token
agreement, directory size, selected fraction, page hit rate, bytes loaded, and
the bytes a full document scan would have read.

## Interpretation

This version is an algorithmic prototype, not an end-to-end latency claim. Its
directory construction starts from a full prefill and its Python page copies
are synchronous at the attention boundary. A positive result is preservation
of answer quality with a small selected fraction and high cross-token page hit
rate. The next system step is to build the directory when LMCache stores a
document, feed CacheBlend-corrected pages into the same selector, and replace
the Python attention path with an online-softmax vLLM kernel plus asynchronous
layer-ahead prefetch.
