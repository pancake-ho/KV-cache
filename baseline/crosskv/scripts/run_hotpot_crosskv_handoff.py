from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

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
    clean_short_answer,
)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_memory_gib() -> dict[str, float]:
    return {
        "allocated_gib":
            torch.cuda.memory_allocated()
            / 1024**3,

        "reserved_gib":
            torch.cuda.memory_reserved()
            / 1024**3,

        "peak_allocated_gib":
            torch.cuda.max_memory_allocated()
            / 1024**3,
    }


def clean_prediction(text: str) -> str:
    """
    Keep HotpotQA evaluation robust if Qwen emits
    a thinking block before the final answer.
    """

    text = text.strip()

    if "</think>" in text:
        text = text.split(
            "</think>",
            1,
        )[1].strip()

    return clean_short_answer(text)


@torch.inference_mode()
def measure_target_full_prefill(
    target,
    input_ids: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    """
    Baseline switch cost:

        target receives the complete history
        and performs a full prefill.

    Only the final-token LM-head computation is needed
    for generation.
    """

    device = (
        target.model
        .embed_tokens
        .weight
        .device
    )

    ids = input_ids.to(device)

    attention_mask = torch.ones_like(
        ids,
        dtype=torch.long,
    )

    torch.cuda.reset_peak_memory_stats()

    synchronize()

    start = time.perf_counter()

    output = target.model(
        input_ids=ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    # Generation needs the logits following
    # the final context token.
    _ = target.lm_head(
        output.last_hidden_state[:, -1:, :]
    )

    synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    memory = gpu_memory_gib()

    del output

    gc.collect()
    torch.cuda.empty_cache()

    return elapsed, memory


@torch.inference_mode()
def build_source_cache(
    source,
    input_ids: torch.Tensor,
):
    """
    Source has already processed the history in the
    intended handoff scenario.

    We intentionally stop one token before the end.
    The target handles the final history token itself.
    """

    device = (
        source.model
        .embed_tokens
        .weight
        .device
    )

    source_prefix = (
        input_ids[:, :-1]
        .to(device)
    )

    attention_mask = torch.ones_like(
        source_prefix,
        dtype=torch.long,
    )

    position_ids = (
        attention_mask
        .cumsum(dim=1)
        .sub(1)
    )

    torch.cuda.reset_peak_memory_stats()

    synchronize()

    start = time.perf_counter()

    output = source.model(
        input_ids=source_prefix,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )

    synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    memory = gpu_memory_gib()

    return (
        output.past_key_values,
        elapsed,
        memory,
    )


@torch.inference_mode()
def measure_crosskv_switch(
    *,
    source,
    target,
    mapper,
    source_cache,
    input_ids: torch.Tensor,
) -> tuple[
    float,
    float,
    float,
    dict[str, float],
]:
    """
    Actual switch path:

        existing source KV
            ->
        CrossKV mapping
            ->
        target processes one bridge token

    Source-prefill time is NOT included.
    """

    target_device = (
        target.model
        .embed_tokens
        .weight
        .device
    )

    context_length = (
        input_ids.shape[1]
    )

    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------
    # A. KV mapping
    # --------------------------------------------------

    synchronize()

    map_start = time.perf_counter()

    mapped_cache = mapper.map_cache(
        source_cache,
        source_model=source,
        target_model=target,
    )

    synchronize()

    map_seconds = (
        time.perf_counter()
        - map_start
    )

    # --------------------------------------------------
    # B. Target bridge token
    #
    # Source cache contains [0, L-2].
    # Target processes context token L-1 itself.
    # --------------------------------------------------

    bridge_token = (
        input_ids[:, -1:]
        .to(target_device)
    )

    past_length = (
        context_length - 1
    )

    attention_mask = torch.ones(
        (
            1,
            context_length,
        ),
        dtype=torch.long,
        device=target_device,
    )

    position_ids = torch.tensor(
        [[past_length]],
        dtype=torch.long,
        device=target_device,
    )

    cache_position = torch.tensor(
        [past_length],
        dtype=torch.long,
        device=target_device,
    )

    synchronize()

    bridge_start = time.perf_counter()

    bridge_output = target.model(
        input_ids=bridge_token,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=mapped_cache,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )

    _ = target.lm_head(
        bridge_output.last_hidden_state[:, -1:, :]
    )

    synchronize()

    bridge_seconds = (
        time.perf_counter()
        - bridge_start
    )

    total_seconds = (
        map_seconds
        + bridge_seconds
    )

    memory = gpu_memory_gib()

    del bridge_output
    del mapped_cache

    gc.collect()
    torch.cuda.empty_cache()

    return (
        map_seconds,
        bridge_seconds,
        total_seconds,
        memory,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

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
        "--prompt-file",
        required=True,
    )

    parser.add_argument(
        "--meta-file",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )

    args = parser.parse_args()


    if not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: CUDA unavailable. "
            "CPU fallback is not permitted."
        )


    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "GPU total GiB:",
        torch.cuda.get_device_properties(
            0
        ).total_memory / 1024**3,
    )


    prompt = Path(
        args.prompt_file
    ).read_text(
        encoding="utf-8"
    )

    meta = json.loads(
        Path(
            args.meta_file
        ).read_text(
            encoding="utf-8"
        )
    )

    golds = [
        str(x)
        for x in meta["answers"]
    ]


    # ==================================================
    # Tokenizers
    # ==================================================

    print()
    print(
        "Loading tokenizers...",
        flush=True,
    )

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
            "Source and target tokenizer IDs differ. "
            "Cross-model handoff requires identical "
            "history tokenization."
        )


    input_ids = source_ids

    context_tokens = int(
        input_ids.shape[1]
    )


    print(
        "Context tokens:",
        context_tokens,
    )

    print(
        "Question:",
        meta["question"],
    )

    print(
        "Gold answers:",
        golds,
    )


    if (
        "prompt_tokens" in meta
        and int(meta["prompt_tokens"])
        != context_tokens
    ):
        raise RuntimeError(
            "Stored prompt token count does not "
            "match runtime tokenization."
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
            device_map={"": "cuda:0"},
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
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        )
        .eval()
    )


    print()
    print(
        "GPU memory after models:",
        gpu_memory_gib(),
    )


    # ==================================================
    # Mapper
    # ==================================================

    mapper = LinearKVMapper(
        args.mapper
    )


    if (
        mapper.config.source_model
        != args.source_model
    ):
        raise RuntimeError(
            "Mapper source-model mismatch."
        )


    if (
        mapper.config.target_model
        != args.target_model
    ):
        raise RuntimeError(
            "Mapper target-model mismatch."
        )


    print()
    print(
        "Preloading mapper onto target GPU...",
        flush=True,
    )


    synchronize()

    mapper_load_start = (
        time.perf_counter()
    )


    mapper.preload_for_model(
        target
    )


    synchronize()


    mapper_preload_seconds = (
        time.perf_counter()
        - mapper_load_start
    )


    print(
        "Mapper preload seconds:",
        mapper_preload_seconds,
    )

    print(
        "GPU memory after mapper:",
        gpu_memory_gib(),
    )


    # ==================================================
    # Minimal warm-up
    # ==================================================

    print()
    print(
        "Warm-up...",
        flush=True,
    )


    warm = input_ids[:, :32].to(
        "cuda:0"
    )


    with torch.inference_mode():

        _ = source.model(
            input_ids=warm,
            use_cache=False,
            return_dict=True,
        )

        _ = target.model(
            input_ids=warm,
            use_cache=False,
            return_dict=True,
        )


    synchronize()

    gc.collect()
    torch.cuda.empty_cache()


    # ==================================================
    # Baseline target full-prefill
    # ==================================================

    print()
    print(
        "Measuring target full prefill...",
        flush=True,
    )


    (
        target_prefill_seconds,
        target_prefill_memory,
    ) = measure_target_full_prefill(
        target,
        input_ids,
    )


    # ==================================================
    # Source cache
    # ==================================================

    print()
    print(
        "Building source KV cache...",
        flush=True,
    )


    (
        source_cache,
        source_prefill_seconds,
        source_prefill_memory,
    ) = build_source_cache(
        source,
        input_ids,
    )


    # ==================================================
    # Actual CrossKV switch
    # ==================================================

    print()
    print(
        "Running CrossKV switch...",
        flush=True,
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
        input_ids=input_ids,
    )


    # We no longer need the timing cache.
    del source_cache

    gc.collect()
    torch.cuda.empty_cache()


    # ==================================================
    # Actual generations
    #
    # Use the repository's existing, tested
    # generation path rather than duplicating it.
    # ==================================================

    print()
    print(
        "Generating native and transferred answers...",
        flush=True,
    )


    synchronize()

    generation_start = (
        time.perf_counter()
    )


    pair = generate_pair(
        source_model=source,
        target_model=target,
        mapper=mapper,
        tokenizer=source_tokenizer,
        context=prompt,
        max_new_tokens=args.max_new_tokens,
        stop=(),
    )


    synchronize()


    generation_pair_seconds = (
        time.perf_counter()
        - generation_start
    )


    native_clean = clean_prediction(
        pair.standalone
    )

    transfer_clean = clean_prediction(
        pair.transfer
    )


    native_em = answer_em(
        native_clean,
        golds,
    )

    native_f1 = answer_f1(
        native_clean,
        golds,
    )

    transfer_em = answer_em(
        transfer_clean,
        golds,
    )

    transfer_f1 = answer_f1(
        transfer_clean,
        golds,
    )


    speedup = (
        target_prefill_seconds
        / switch_seconds
        if switch_seconds > 0
        else None
    )


    result = {
        "sample": {
            "index":
                meta.get("index"),

            "id":
                meta.get("id"),

            "question":
                meta["question"],

            "gold_answers":
                golds,

            "context_tokens":
                context_tokens,
        },

        "models": {
            "source":
                args.source_model,

            "target":
                args.target_model,

            "mapper":
                args.mapper,

            "mapper_calibration_observations":
                mapper.config.num_observations,

            "mapper_ridge":
                mapper.config.ridge,

            "mapper_top_k":
                len(
                    mapper.config
                    .selected_layers[0]
                ),
        },

        "timing_seconds": {
            "mapper_preload_cold_setup":
                mapper_preload_seconds,

            "target_full_prefill":
                target_prefill_seconds,

            "source_prefill_existing_work":
                source_prefill_seconds,

            "crosskv_map":
                map_seconds,

            "target_bridge_one_token":
                bridge_seconds,

            "crosskv_switch_total":
                switch_seconds,

            "switch_speedup_vs_target_reprefill":
                speedup,

            "generation_pair_total":
                generation_pair_seconds,
        },

        "memory": {
            "target_full_prefill":
                target_prefill_memory,

            "source_prefill":
                source_prefill_memory,

            "crosskv_switch":
                switch_memory,
        },

        "native_target": {
            "raw":
                pair.standalone,

            "clean":
                native_clean,

            "generated_tokens":
                pair.standalone_tokens,

            "em":
                native_em,

            "f1":
                native_f1,
        },

        "crosskv_transfer": {
            "raw":
                pair.transfer,

            "clean":
                transfer_clean,

            "generated_tokens":
                pair.transfer_tokens,

            "em":
                transfer_em,

            "f1":
                transfer_f1,
        },
    }


    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
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
        "REAL CROSSKV HANDOFF RESULT"
    )
    print(
        "=" * 70
    )

    print(
        f"Context tokens       : "
        f"{context_tokens}"
    )

    print(
        f"Gold                 : "
        f"{golds}"
    )

    print()

    print(
        f"Target full prefill   : "
        f"{target_prefill_seconds:.6f} s"
    )

    print(
        f"Source prefill        : "
        f"{source_prefill_seconds:.6f} s "
        f"(existing work; excluded from switch)"
    )

    print(
        f"CrossKV mapping       : "
        f"{map_seconds:.6f} s"
    )

    print(
        f"Target 1-token bridge : "
        f"{bridge_seconds:.6f} s"
    )

    print(
        f"CrossKV switch total  : "
        f"{switch_seconds:.6f} s"
    )

    print(
        f"Switch speedup        : "
        f"{speedup:.3f}x"
    )

    print()

    print(
        "[Native 4B]"
    )

    print(
        pair.standalone
    )

    print(
        "clean =",
        native_clean,
    )

    print(
        "EM/F1 =",
        native_em,
        native_f1,
    )

    print()

    print(
        "[CrossKV 1.7B -> 4B]"
    )

    print(
        pair.transfer
    )

    print(
        "clean =",
        transfer_clean,
    )

    print(
        "EM/F1 =",
        transfer_em,
        transfer_f1,
    )

    print()

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()