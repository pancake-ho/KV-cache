from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from xmodel_kv.artifact import LinearKVMapper
from xmodel_kv.generation import generate_pair
from xmodel_kv.hotpotqa import (
    answer_em,
    answer_f1,
)

from run_hotpot_crosskv_handoff import (
    build_source_cache,
    clean_prediction,
    measure_crosskv_switch,
    measure_target_full_prefill,
    synchronize,
)


def build_prompt(row: dict) -> str:
    return (
        "Answer the question using only the information "
        "in the context below.\n"
        "Give only a short final answer.\n\n"
        "=== CONTEXT ===\n"
        f"{row['context']}\n"
        "=== END CONTEXT ===\n\n"
        f"Question: {row['input']}\n"
        "Answer:"
    )


@torch.inference_mode()
def generate_source_native(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:

    device = (
        model.model
        .embed_tokens
        .weight
        .device
    )

    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )

    input_ids = encoded.input_ids.to(
        device
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )

    generated = output[
        0,
        input_ids.shape[1]:,
    ]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    del output
    del input_ids
    del attention_mask

    gc.collect()
    torch.cuda.empty_cache()

    return text


def load_dataset(path: Path) -> list[dict]:
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:

        for index, line in enumerate(f):

            row = json.loads(line)

            row["_local_index"] = index

            rows.append(row)

    return rows


def choose_samples(
    *,
    rows: list[dict],
    tokenizer,
    target_lengths: list[int],
    per_length: int,
) -> list[dict]:

    candidates = []

    for row in rows:

        prompt = build_prompt(row)

        token_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        candidates.append({
            "index":
                row["_local_index"],

            "id":
                row.get("_id"),

            "question":
                row["input"],

            "answers":
                row["answers"],

            "context_tokens":
                len(token_ids),

            "prompt":
                prompt,
        })

    selected = []
    used_indices = set()

    for target_length in target_lengths:

        ranked = sorted(
            candidates,
            key=lambda x:
                abs(
                    x["context_tokens"]
                    - target_length
                ),
        )

        bucket_count = 0

        for candidate in ranked:

            if (
                candidate["index"]
                in used_indices
            ):
                continue

            item = dict(candidate)

            item["target_length"] = (
                target_length
            )

            selected.append(item)

            used_indices.add(
                candidate["index"]
            )

            bucket_count += 1

            if (
                bucket_count
                >= per_length
            ):
                break

        if bucket_count < per_length:
            raise RuntimeError(
                f"Could only select "
                f"{bucket_count} samples "
                f"near target length "
                f"{target_length}"
            )

    return selected


def completed_indices(
    output_path: Path,
) -> set[int]:

    if not output_path.exists():
        return set()

    completed = set()

    with output_path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            completed.add(
                int(row["index"])
            )

    return completed


def summarize(
    rows: list[dict],
) -> dict:

    grouped = defaultdict(list)

    for row in rows:
        grouped[
            int(row["target_length"])
        ].append(row)

    summary = {
        "documents":
            len(rows),

        "overall": {},
        "by_target_length": {},
    }

    def aggregate(
        subset: list[dict],
    ) -> dict:

        n = len(subset)

        source_em = np.asarray(
            [
                x["source_native"]["em"]
                for x in subset
            ],
            dtype=np.float64,
        )

        target_em = np.asarray(
            [
                x["target_native"]["em"]
                for x in subset
            ],
            dtype=np.float64,
        )

        transfer_em = np.asarray(
            [
                x["crosskv_transfer"]["em"]
                for x in subset
            ],
            dtype=np.float64,
        )

        prefill = np.asarray(
            [
                x["timing_seconds"][
                    "target_full_prefill"
                ]
                for x in subset
            ]
        )

        mapping = np.asarray(
            [
                x["timing_seconds"][
                    "crosskv_map"
                ]
                for x in subset
            ]
        )

        bridge = np.asarray(
            [
                x["timing_seconds"][
                    "target_bridge_one_token"
                ]
                for x in subset
            ]
        )

        switch = np.asarray(
            [
                x["timing_seconds"][
                    "crosskv_switch_total"
                ]
                for x in subset
            ]
        )

        speedup = (
            prefill / switch
        )

        target_correct_rows = [
            x
            for x in subset
            if x["target_native"]["em"]
            == 1.0
        ]

        target_correct = len(
            target_correct_rows
        )

        retained = sum(
            1
            for x in target_correct_rows
            if x["crosskv_transfer"]["em"]
            == 1.0
        )

        lost = (
            target_correct
            - retained
        )

        source_wrong_target_correct = sum(
            1
            for x in subset
            if (
                x["source_native"]["em"]
                == 0.0
                and
                x["target_native"]["em"]
                == 1.0
            )
        )

        source_wrong_target_correct_transfer_wrong = sum(
            1
            for x in subset
            if (
                x["source_native"]["em"]
                == 0.0
                and
                x["target_native"]["em"]
                == 1.0
                and
                x["crosskv_transfer"]["em"]
                == 0.0
            )
        )

        return {
            "n":
                n,

            "mean_context_tokens":
                float(
                    np.mean(
                        [
                            x[
                                "context_tokens"
                            ]
                            for x in subset
                        ]
                    )
                ),

            "source_native_em":
                float(
                    source_em.mean()
                ),

            "target_native_em":
                float(
                    target_em.mean()
                ),

            "crosskv_em":
                float(
                    transfer_em.mean()
                ),

            "target_correct":
                target_correct,

            "crosskv_retained_target_correct":
                retained,

            "crosskv_lost_target_correct":
                lost,

            "conditional_retention":
                (
                    retained
                    / target_correct
                    if target_correct
                    else None
                ),

            "source_wrong_target_correct":
                source_wrong_target_correct,

            "source_wrong_target_correct_transfer_wrong":
                (
                    source_wrong_target_correct_transfer_wrong
                ),

            "target_full_prefill_mean_s":
                float(
                    prefill.mean()
                ),

            "crosskv_map_mean_s":
                float(
                    mapping.mean()
                ),

            "bridge_mean_s":
                float(
                    bridge.mean()
                ),

            "crosskv_switch_mean_s":
                float(
                    switch.mean()
                ),

            "speedup_mean":
                float(
                    speedup.mean()
                ),

            "speedup_median":
                float(
                    np.median(
                        speedup
                    )
                ),

            "crosskv_faster_count":
                int(
                    np.sum(
                        speedup > 1.0
                    )
                ),
        }

    summary["overall"] = aggregate(
        rows
    )

    for target_length, subset in sorted(
        grouped.items()
    ):
        summary[
            "by_target_length"
        ][str(target_length)] = aggregate(
            subset
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--source-model",
        default="Qwen/Qwen3-1.7B",
    )

    parser.add_argument(
        "--target-model",
        default="Qwen/Qwen3-4B",
    )

    parser.add_argument(
        "--mapper",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--summary",
        required=True,
    )

    parser.add_argument(
        "--selection-output",
        required=True,
    )

    parser.add_argument(
        "--target-lengths",
        nargs="+",
        type=int,
        default=[
            2200,
            4096,
            8192,
        ],
    )

    parser.add_argument(
        "--per-length",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA unavailable. "
            "CPU fallback is forbidden."
        )

    dataset_path = Path(
        args.dataset
    )

    output_path = Path(
        args.output
    )

    summary_path = Path(
        args.summary
    )

    selection_path = Path(
        args.selection_output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    # --------------------------------------------------
    # Tokenizers
    # --------------------------------------------------

    source_tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.source_model
        )
    )

    target_tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.target_model
        )
    )

    rows = load_dataset(
        dataset_path
    )

    selected = choose_samples(
        rows=rows,
        tokenizer=source_tokenizer,
        target_lengths=args.target_lengths,
        per_length=args.per_length,
    )

    selection_path.write_text(
        json.dumps(
            [
                {
                    k: v
                    for k, v in item.items()
                    if k != "prompt"
                }
                for item in selected
            ],
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Selected samples:"
    )

    for item in selected:
        print(
            f"bucket={item['target_length']:5d} "
            f"index={item['index']:4d} "
            f"tokens={item['context_tokens']:5d} "
            f"Q={item['question'][:70]}"
        )

    # --------------------------------------------------
    # Models
    # --------------------------------------------------

    print()
    print(
        "Loading source model...",
        flush=True,
    )

    source = (
        AutoModelForCausalLM
        .from_pretrained(
            args.source_model,
            dtype=torch.bfloat16,
            device_map={
                "":
                    "cuda:0"
            },
            attn_implementation="sdpa",
        )
        .eval()
    )

    print(
        "Loading target model...",
        flush=True,
    )

    target = (
        AutoModelForCausalLM
        .from_pretrained(
            args.target_model,
            dtype=torch.bfloat16,
            device_map={
                "":
                    "cuda:0"
            },
            attn_implementation="sdpa",
        )
        .eval()
    )

    mapper = LinearKVMapper(
        args.mapper
    )

    if (
        mapper.config.source_model
        != args.source_model
    ):
        raise RuntimeError(
            "mapper source-model mismatch"
        )

    if (
        mapper.config.target_model
        != args.target_model
    ):
        raise RuntimeError(
            "mapper target-model mismatch"
        )

    print(
        "Preloading mapper...",
        flush=True,
    )

    mapper.preload_for_model(
        target
    )

    synchronize()

    # --------------------------------------------------
    # Warm up
    # --------------------------------------------------

    warm_ids = source_tokenizer(
        "Warm up.",
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(
        "cuda:0"
    )

    with torch.inference_mode():
        source(
            input_ids=warm_ids,
            use_cache=False,
        )

        target(
            input_ids=warm_ids,
            use_cache=False,
        )

    synchronize()

    del warm_ids

    gc.collect()
    torch.cuda.empty_cache()

    completed = completed_indices(
        output_path
    )

    # --------------------------------------------------
    # Real samples
    # --------------------------------------------------

    for sample_number, item in enumerate(
        selected,
        start=1,
    ):

        if item["index"] in completed:
            print(
                f"[{sample_number}/{len(selected)}] "
                f"index={item['index']} "
                f"already complete -> skip"
            )
            continue

        print()
        print(
            "=" * 70
        )

        print(
            f"[{sample_number}/{len(selected)}] "
            f"index={item['index']} "
            f"tokens={item['context_tokens']} "
            f"bucket={item['target_length']}"
        )

        print(
            "=" * 70
        )

        prompt = item["prompt"]

        golds = [
            str(x)
            for x in item[
                "answers"
            ]
        ]

        source_ids = source_tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids

        target_ids = target_tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids

        if not torch.equal(
            source_ids,
            target_ids,
        ):
            raise RuntimeError(
                "source/target tokenization mismatch"
            )

        # ----------------------------------------------
        # Target re-prefill timing
        # ----------------------------------------------

        (
            target_prefill,
            target_memory,
        ) = measure_target_full_prefill(
            target,
            source_ids,
        )

        # ----------------------------------------------
        # Existing source cache + switch timing
        # ----------------------------------------------

        (
            source_cache,
            source_prefill,
            source_memory,
        ) = build_source_cache(
            source,
            source_ids,
        )

        (
            map_seconds,
            bridge_seconds,
            switch_seconds,
            switch_memory,
        ) = measure_crosskv_switch(
            source=source,
            target=target,
            mapper=mapper,
            source_cache=source_cache,
            input_ids=source_ids,
        )

        del source_cache

        gc.collect()
        torch.cuda.empty_cache()

        # ----------------------------------------------
        # Source native
        # ----------------------------------------------

        source_raw = (
            generate_source_native(
                model=source,
                tokenizer=source_tokenizer,
                prompt=prompt,
                max_new_tokens=(
                    args.max_new_tokens
                ),
            )
        )

        source_clean = (
            clean_prediction(
                source_raw
            )
        )

        # ----------------------------------------------
        # Target native + CrossKV continuation
        # ----------------------------------------------

        pair = generate_pair(
            source_model=source,
            target_model=target,
            mapper=mapper,
            tokenizer=source_tokenizer,
            context=prompt,
            max_new_tokens=(
                args.max_new_tokens
            ),
            stop=(),
        )

        target_clean = clean_prediction(
            pair.standalone
        )

        transfer_clean = clean_prediction(
            pair.transfer
        )

        row = {
            "index":
                item["index"],

            "id":
                item["id"],

            "target_length":
                item[
                    "target_length"
                ],

            "context_tokens":
                item[
                    "context_tokens"
                ],

            "question":
                item["question"],

            "gold_answers":
                golds,

            "source_native": {
                "raw":
                    source_raw,

                "clean":
                    source_clean,

                "em":
                    answer_em(
                        source_clean,
                        golds,
                    ),

                "f1":
                    answer_f1(
                        source_clean,
                        golds,
                    ),
            },

            "target_native": {
                "raw":
                    pair.standalone,

                "clean":
                    target_clean,

                "em":
                    answer_em(
                        target_clean,
                        golds,
                    ),

                "f1":
                    answer_f1(
                        target_clean,
                        golds,
                    ),
            },

            "crosskv_transfer": {
                "raw":
                    pair.transfer,

                "clean":
                    transfer_clean,

                "em":
                    answer_em(
                        transfer_clean,
                        golds,
                    ),

                "f1":
                    answer_f1(
                        transfer_clean,
                        golds,
                    ),
            },

            "timing_seconds": {
                "target_full_prefill":
                    target_prefill,

                "source_prefill_existing_work":
                    source_prefill,

                "crosskv_map":
                    map_seconds,

                "target_bridge_one_token":
                    bridge_seconds,

                "crosskv_switch_total":
                    switch_seconds,

                "speedup":
                    (
                        target_prefill
                        / switch_seconds
                    ),
            },

            "memory": {
                "target_full_prefill":
                    target_memory,

                "source_prefill":
                    source_memory,

                "crosskv_switch":
                    switch_memory,
            },
        }

        with output_path.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

        print(
            "Gold      :",
            golds,
        )

        print(
            "Source    :",
            source_clean,
            "| EM=",
            row[
                "source_native"
            ]["em"],
        )

        print(
            "Target    :",
            target_clean,
            "| EM=",
            row[
                "target_native"
            ]["em"],
        )

        print(
            "CrossKV   :",
            transfer_clean,
            "| EM=",
            row[
                "crosskv_transfer"
            ]["em"],
        )

        print(
            f"Prefill={target_prefill:.4f}s "
            f"Map={map_seconds:.4f}s "
            f"Bridge={bridge_seconds:.4f}s "
            f"Switch={switch_seconds:.4f}s "
            f"Speedup="
            f"{target_prefill/switch_seconds:.3f}x"
        )

        del source_ids
        del target_ids
        del pair

        gc.collect()
        torch.cuda.empty_cache()

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    result_rows = []

    with output_path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():
                result_rows.append(
                    json.loads(line)
                )

    summary = summarize(
        result_rows
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=" * 70
    )
    print(
        "SUMMARY"
    )
    print(
        "=" * 70
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
