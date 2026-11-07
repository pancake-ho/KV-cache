from __future__ import annotations

import argparse
import gc
import json
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

from run_hotpot_crosskv_batch import (
    build_prompt,
    generate_source_native,
    load_dataset,
)

from run_hotpot_crosskv_handoff import (
    build_source_cache,
    clean_prediction,
    measure_crosskv_switch,
    measure_target_full_prefill,
    synchronize,
)


def completed_indices(
    path: Path,
) -> set[int]:

    if not path.exists():
        return set()

    result = set()

    with path.open(
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            result.add(
                int(row["index"])
            )

    return result


def assign_bin(
    tokens: int,
) -> str:

    if tokens < 3000:
        return "0-3k"

    if tokens < 5000:
        return "3-5k"

    if tokens < 8000:
        return "5-8k"

    return "8-10k"


def aggregate(
    rows: list[dict],
) -> dict:

    if not rows:
        return {
            "n": 0,
        }

    n = len(rows)

    source_em = np.asarray(
        [
            x["source_native"]["em"]
            for x in rows
        ],
        dtype=np.float64,
    )

    target_em = np.asarray(
        [
            x["target_native"]["em"]
            for x in rows
        ],
        dtype=np.float64,
    )

    transfer_em = np.asarray(
        [
            x["crosskv_transfer"]["em"]
            for x in rows
        ],
        dtype=np.float64,
    )

    source_f1 = np.asarray(
        [
            x["source_native"]["f1"]
            for x in rows
        ],
        dtype=np.float64,
    )

    target_f1 = np.asarray(
        [
            x["target_native"]["f1"]
            for x in rows
        ],
        dtype=np.float64,
    )

    transfer_f1 = np.asarray(
        [
            x["crosskv_transfer"]["f1"]
            for x in rows
        ],
        dtype=np.float64,
    )

    prefill = np.asarray(
        [
            x["timing_seconds"][
                "target_full_prefill"
            ]
            for x in rows
        ],
        dtype=np.float64,
    )

    mapping = np.asarray(
        [
            x["timing_seconds"][
                "crosskv_map"
            ]
            for x in rows
        ],
        dtype=np.float64,
    )

    bridge = np.asarray(
        [
            x["timing_seconds"][
                "target_bridge_one_token"
            ]
            for x in rows
        ],
        dtype=np.float64,
    )

    switch = np.asarray(
        [
            x["timing_seconds"][
                "crosskv_switch_total"
            ]
            for x in rows
        ],
        dtype=np.float64,
    )

    speedup = (
        prefill
        / switch
    )

    saved_fraction = (
        1.0
        - switch / prefill
    )

    target_correct_rows = [
        x
        for x in rows
        if (
            x["target_native"]["em"]
            == 1.0
        )
    ]

    target_correct = len(
        target_correct_rows
    )

    retained = sum(
        1
        for x in target_correct_rows
        if (
            x["crosskv_transfer"]["em"]
            == 1.0
        )
    )

    escalation_rows = [
        x
        for x in rows
        if (
            x["source_native"]["em"]
            == 0.0
            and
            x["target_native"]["em"]
            == 1.0
        )
    ]

    escalation_cases = len(
        escalation_rows
    )

    escalation_recovered = sum(
        1
        for x in escalation_rows
        if (
            x["crosskv_transfer"]["em"]
            == 1.0
        )
    )

    # Both source and target can solve it,
    # but transferred cache causes failure.
    mapper_damage_rows = [
        x
        for x in rows
        if (
            x["source_native"]["em"]
            == 1.0
            and
            x["target_native"]["em"]
            == 1.0
            and
            x["crosskv_transfer"]["em"]
            == 0.0
        )
    ]

    crosskv_only_correct = sum(
        1
        for x in rows
        if (
            x["target_native"]["em"]
            == 0.0
            and
            x["crosskv_transfer"]["em"]
            == 1.0
        )
    )

    target_peak = np.asarray(
        [
            x["memory"][
                "target_full_prefill"
            ][
                "peak_allocated_gib"
            ]
            for x in rows
        ],
        dtype=np.float64,
    )

    switch_peak = np.asarray(
        [
            x["memory"][
                "crosskv_switch"
            ][
                "peak_allocated_gib"
            ]
            for x in rows
        ],
        dtype=np.float64,
    )

    return {
        "n":
            n,

        "mean_context_tokens":
            float(
                np.mean(
                    [
                        x["context_tokens"]
                        for x in rows
                    ]
                )
            ),

        "source_native_em":
            float(source_em.mean()),

        "target_native_em":
            float(target_em.mean()),

        "crosskv_em":
            float(transfer_em.mean()),

        "source_native_f1":
            float(source_f1.mean()),

        "target_native_f1":
            float(target_f1.mean()),

        "crosskv_f1":
            float(transfer_f1.mean()),

        "target_correct":
            target_correct,

        "crosskv_retained_target_correct":
            retained,

        "crosskv_lost_target_correct":
            target_correct - retained,

        "conditional_target_retention":
            (
                retained
                / target_correct
                if target_correct > 0
                else None
            ),

        "source_wrong_target_correct":
            escalation_cases,

        "source_wrong_target_correct_crosskv_correct":
            escalation_recovered,

        "escalation_recovery_rate":
            (
                escalation_recovered
                / escalation_cases
                if escalation_cases > 0
                else None
            ),

        "source_correct_target_correct_crosskv_wrong":
            len(
                mapper_damage_rows
            ),

        "target_wrong_crosskv_correct":
            crosskv_only_correct,

        "target_full_prefill_mean_s":
            float(prefill.mean()),

        "target_full_prefill_median_s":
            float(
                np.median(prefill)
            ),

        "crosskv_map_mean_s":
            float(mapping.mean()),

        "crosskv_bridge_mean_s":
            float(bridge.mean()),

        "crosskv_switch_mean_s":
            float(switch.mean()),

        "crosskv_switch_median_s":
            float(
                np.median(switch)
            ),

        "speedup_mean":
            float(speedup.mean()),

        "speedup_median":
            float(
                np.median(speedup)
            ),

        "latency_saved_percent_mean":
            float(
                100.0
                * saved_fraction.mean()
            ),

        "crosskv_faster_count":
            int(
                np.sum(
                    speedup > 1.0
                )
            ),

        "crosskv_faster_fraction":
            float(
                np.mean(
                    speedup > 1.0
                )
            ),

        "mapper_fraction_of_switch_mean":
            float(
                np.mean(
                    mapping / switch
                )
            ),

        "target_prefill_peak_allocated_gib_mean":
            float(
                target_peak.mean()
            ),

        "crosskv_switch_peak_allocated_gib_mean":
            float(
                switch_peak.mean()
            ),
    }


def make_summary(
    rows: list[dict],
    *,
    total_dataset_rows: int,
    max_context_tokens: int,
) -> dict:

    bins = {
        "0-3k": [],
        "3-5k": [],
        "5-8k": [],
        "8-10k": [],
    }

    for row in rows:

        bins[
            assign_bin(
                int(
                    row[
                        "context_tokens"
                    ]
                )
            )
        ].append(row)

    return {
        "total_dataset_rows":
            total_dataset_rows,

        "max_context_tokens":
            max_context_tokens,

        "evaluated_documents":
            len(rows),

        "overall":
            aggregate(rows),

        "by_context_bin": {
            name:
                aggregate(bucket)
            for name, bucket
            in bins.items()
        },
    }


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
        "--eligible-output",
        required=True,
    )

    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=10000,
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

    eligible_path = Path(
        args.eligible_output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # Dataset / tokenizer
    # ==================================================

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

    raw_rows = load_dataset(
        dataset_path
    )

    eligible = []

    print(
        "Scanning token lengths..."
    )

    for row in raw_rows:

        prompt = build_prompt(row)

        token_ids = source_tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        context_tokens = len(
            token_ids
        )

        if (
            context_tokens
            > args.max_context_tokens
        ):
            continue

        eligible.append({
            "index":
                int(
                    row[
                        "_local_index"
                    ]
                ),

            "id":
                row.get("_id"),

            "question":
                row["input"],

            "answers":
                row["answers"],

            "context_tokens":
                context_tokens,

            "prompt":
                prompt,
        })

    # Shortest -> longest.
    eligible.sort(
        key=lambda x:
            x["context_tokens"]
    )

    eligible_path.write_text(
        json.dumps(
            [
                {
                    k: v
                    for k, v in item.items()
                    if k != "prompt"
                }
                for item in eligible
            ],
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        "Dataset rows:",
        len(raw_rows),
    )

    print(
        "Eligible <=",
        args.max_context_tokens,
        ":",
        len(eligible),
    )

    if not eligible:
        raise RuntimeError(
            "No eligible samples."
        )

    print(
        "Eligible token range:",
        eligible[0]["context_tokens"],
        "to",
        eligible[-1]["context_tokens"],
    )

    # ==================================================
    # Models
    # ==================================================

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
            "Mapper source mismatch."
        )

    if (
        mapper.config.target_model
        != args.target_model
    ):
        raise RuntimeError(
            "Mapper target mismatch."
        )

    print(
        "Preloading mapper...",
        flush=True,
    )

    mapper.preload_for_model(
        target
    )

    synchronize()

    # ==================================================
    # Warm-up
    # ==================================================

    warm = source_tokenizer(
        "Warm up.",
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(
        "cuda:0"
    )

    with torch.inference_mode():

        _ = source(
            input_ids=warm,
            use_cache=False,
        )

        _ = target(
            input_ids=warm,
            use_cache=False,
        )

    synchronize()

    del warm

    gc.collect()
    torch.cuda.empty_cache()

    completed = completed_indices(
        output_path
    )

    print(
        "Already completed:",
        len(completed),
    )

    # ==================================================
    # Evaluation
    # ==================================================

    for number, item in enumerate(
        eligible,
        start=1,
    ):

        index = item["index"]

        if index in completed:

            print(
                f"[{number}/{len(eligible)}] "
                f"index={index} "
                f"already complete"
            )

            continue

        print()
        print(
            "=" * 72
        )

        print(
            f"[{number}/{len(eligible)}] "
            f"index={index} "
            f"tokens="
            f"{item['context_tokens']}"
        )

        print(
            "=" * 72
        )

        prompt = item["prompt"]

        golds = [
            str(x)
            for x in item["answers"]
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
                "Source/target tokenizer mismatch."
            )

        # ----------------------------------------------
        # Baseline target re-prefill timing
        # ----------------------------------------------

        (
            target_prefill,
            target_memory,
        ) = measure_target_full_prefill(
            target,
            source_ids,
        )

        # ----------------------------------------------
        # Existing source cache + transfer timing
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
        # Source native answer
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
        # Target native + CrossKV
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

        target_clean = (
            clean_prediction(
                pair.standalone
            )
        )

        transfer_clean = (
            clean_prediction(
                pair.transfer
            )
        )

        row = {
            "index":
                index,

            "id":
                item["id"],

            "context_tokens":
                item[
                    "context_tokens"
                ],

            "context_bin":
                assign_bin(
                    item[
                        "context_tokens"
                    ]
                ),

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
            "Gold    :",
            golds,
        )

        print(
            "Source  :",
            source_clean,
            "| EM=",
            row[
                "source_native"
            ]["em"],
        )

        print(
            "Target  :",
            target_clean,
            "| EM=",
            row[
                "target_native"
            ]["em"],
        )

        print(
            "CrossKV :",
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

    # ==================================================
    # Final summary
    # ==================================================

    result_rows = []

    with output_path.open(
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            result_rows.append(
                json.loads(line)
            )

    summary = make_summary(
        result_rows,
        total_dataset_rows=len(
            raw_rows
        ),
        max_context_tokens=(
            args.max_context_tokens
        ),
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
        "=" * 72
    )
    print(
        "FULL OPERATING CURVE SUMMARY"
    )
    print(
        "=" * 72
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
