from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..coqa import coqa_f1
from ..policy_imprinting import build_chat_segments, prefill_legacy_cache
from .evaluate_amortized_musique_handoff import _clean_generation
from .probe_semantic_kv_summary import _generate, _normalized


SOURCE_ANSWER_SYSTEM = (
    "You are Agent A, a careful reader. Find the answer to the assigned question "
    "from the supplied dossier. Output only the exact short answer without reasoning."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether source Agent A can solve the MuSiQue first hop."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    if min(args.offset, args.count, args.max_new_tokens) < 0 or not args.count:
        parser.error("offset must be non-negative and counts positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()
    cases = _load_slice(args.dataset, args.offset, args.count)
    rows = []
    for position, (index, case) in enumerate(cases, 1):
        documents = "\n\n".join(
            f"[DOC-{number}] {document['title']}\n{document['text']}"
            for number, document in enumerate(case["agent_a"]["documents"], 1)
        )
        user = (
            f"<DOSSIER>\n{documents}\n</DOSSIER>\n\n"
            f"Question: {case['agent_a']['question']}\n"
            "Output only the exact short answer."
        )
        segments = build_chat_segments(
            tokenizer,
            system_prompt=SOURCE_ANSWER_SYSTEM,
            tools=[],
            history=[{"role": "user", "content": user}],
            enable_thinking=False,
        )
        prefix_ids = list(segments.prefix_ids)
        cache = prefill_legacy_cache(model, prefix_ids)
        source_tokens = len(prefix_ids) + len(segments.history_ids) + len(segments.readout_ids)
        generated = _clean_generation(
            _generate(
                model,
                tokenizer,
                legacy_cache=cache,
                cached_tokens=len(prefix_ids),
                query_ids=(*segments.history_ids, *segments.readout_ids),
                max_new_tokens=args.max_new_tokens,
            )
        )
        gold = case["agent_a"]["gold_answer"]
        normalized_generation = _normalized(generated)
        normalized_gold = _normalized(gold)
        row = {
            "index": index,
            "id": case["id"],
            "source_tokens": source_tokens,
            "generation": generated,
            "gold": gold,
            "exact": float(normalized_generation == normalized_gold),
            "contains": float(normalized_gold in normalized_generation),
            "f1": coqa_f1(generated, [gold]),
        }
        rows.append(row)
        print(json.dumps({"event": "case", "position": position, **row}, ensure_ascii=False), flush=True)
        del cache
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "config": vars(args),
        "cases": len(rows),
        "exact_accuracy": _mean(row["exact"] for row in rows),
        "contains_accuracy": _mean(row["contains"] for row in rows),
        "mean_f1": _mean(row["f1"] for row in rows),
        "mean_source_tokens": _mean(row["source_tokens"] for row in rows),
    }
    with (output_dir / "results.jsonl").open("w") as result_file:
        for row in rows:
            result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)


def _load_slice(path: str, offset: int, count: int):
    rows = []
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            if offset <= index < offset + count:
                rows.append((index, json.loads(line)))
            if index >= offset + count:
                break
    if len(rows) != count:
        raise ValueError("dataset does not contain the requested slice")
    return rows


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
