"""Evaluate real mapped KV continuation quality against the target's native cache."""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .spec import MapperSpec
from .transfer import NeuralKVMapper


def evaluate(*, mapper_dir: str, input_jsonl: str, output_json: str,
             prefix_length: int, max_examples: int = 4, token_chunk: int = 128,
             source_model: str | None = None, target_model: str | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for evaluation")
    if prefix_length < 2 or max_examples < 1:
        raise ValueError("invalid prefix length or number of examples")
    spec = MapperSpec.load(mapper_dir)
    src = source_model or spec.source_model
    tgt = target_model or spec.target_model
    if (src, tgt) != (spec.source_model, spec.target_model):
        raise ValueError("model identifiers differ from mapper calibration metadata")
    mapper = NeuralKVMapper(mapper_dir, token_chunk=token_chunk)
    mapper.assert_complete()
    # Load both models onto the assigned GPU. For this first pair (1.7B -> 4B),
    # users must confirm available VRAM before launch.
    token_s = AutoTokenizer.from_pretrained(src)
    token_t = AutoTokenizer.from_pretrained(tgt)
    model_s = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    model_t = AutoModelForCausalLM.from_pretrained(
        tgt, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    # Existing CrossKV evaluator consumes mapper.map_cache(), preserving the
    # same target-prefix boundary / token-shift protocol as the Ridge baseline.
    from xmodel_kv.evaluation import compare_continuation

    records = []
    with Path(input_jsonl).open(encoding="utf-8") as stream:
        for line_index, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row["text"]
            ids_s = token_s(prompt, add_special_tokens=False)["input_ids"]
            ids_t = token_t(prompt, add_special_tokens=False)["input_ids"]
            if ids_s != ids_t:
                raise ValueError(f"tokenizers produce different input IDs for example {line_index}")
            if len(ids_s) <= prefix_length:
                raise ValueError(f"example {line_index} needs > {prefix_length} tokens")
            input_ids = torch.tensor([ids_s], dtype=torch.long)
            score = compare_continuation(source_model=model_s, target_model=model_t,
                                         mapper=mapper, input_ids=input_ids,
                                         prefix_length=prefix_length)
            records.append({"line_index": line_index, **score.__dict__})
            print(f"[{len(records)}/{max_examples}] NLL delta {score.nll_delta:+.5f}, top1 {score.top1_agreement:.4f}", flush=True)
            if len(records) >= max_examples:
                break
    if not records:
        raise ValueError("input JSONL contains no evaluation examples")
    aggregate = {key: sum(float(item[key]) for item in records)/len(records)
                 for key in ("standalone_nll", "transfer_nll", "nll_delta", "top1_agreement", "logprob_mae")}
    output = {"status": "completed", "scope": "continuation quality ONLY (not prefill/TTFT/E2E benchmark)",
              "source_model": src, "target_model": tgt,
              "mapper_dir": mapper_dir, "prefix_length": prefix_length,
              "num_examples": len(records), "aggregate": aggregate, "records": records,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "python": platform.python_version()}
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(output, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    os.replace(temp, path)
