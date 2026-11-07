from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(line)
                )

    return rows


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


def load_dataset(
    path: Path,
) -> dict[int, dict]:

    result = {}

    with path.open(
        encoding="utf-8"
    ) as f:

        for index, line in enumerate(f):

            row = json.loads(line)

            result[index] = row

    return result


@torch.inference_mode()
def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--results",
        required=True,
    )

    parser.add_argument(
        "--source-model",
        default="Qwen/Qwen3-1.7B",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()


    if not torch.cuda.is_available():

        raise SystemExit(
            "CUDA unavailable. "
            "CPU fallback is forbidden."
        )


    dataset = load_dataset(
        Path(args.dataset)
    )

    results = load_jsonl(
        Path(args.results)
    )


    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.source_model
        )
    )


    print(
        "Loading source model...",
        flush=True,
    )


    model = (
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


    device = (
        model.model
        .embed_tokens
        .weight
        .device
    )


    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Overwrite intentionally:
    # deterministic feature extraction.
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as out:


        for number, result in enumerate(
            sorted(
                results,
                key=lambda x:
                    int(x["index"])
            ),
            start=1,
        ):

            index = int(
                result["index"]
            )

            row = dataset[index]

            prompt = build_prompt(
                row
            )


            encoded = tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )

            input_ids = (
                encoded.input_ids
                .to(device)
            )

            attention_mask = (
                torch.ones_like(
                    input_ids
                )
            )


            context_tokens = int(
                input_ids.shape[1]
            )


            if (
                context_tokens
                != int(
                    result[
                        "context_tokens"
                    ]
                )
            ):

                raise RuntimeError(
                    f"token mismatch "
                    f"index={index}: "
                    f"{context_tokens} vs "
                    f"{result['context_tokens']}"
                )


            question_tokens = len(
                tokenizer(
                    row["input"],
                    add_special_tokens=False,
                )["input_ids"]
            )


            # --------------------------------------------
            # Source backbone only.
            #
            # Do NOT call full causal-LM forward because
            # that would materialize [L, vocab] logits.
            #
            # We only need the final hidden state.
            # --------------------------------------------

            backbone_output = (
                model.model(
                    input_ids=input_ids,
                    attention_mask=(
                        attention_mask
                    ),
                    use_cache=False,
                    return_dict=True,
                )
            )


            last_hidden = (
                backbone_output
                .last_hidden_state[
                    :,
                    -1,
                    :
                ]
                .float()
            )


            logits = (
                model.lm_head(
                    last_hidden
                )
                .float()
            )


            log_probs = (
                torch.log_softmax(
                    logits,
                    dim=-1,
                )
            )

            probs = (
                log_probs.exp()
            )


            entropy = (
                -(
                    probs
                    * log_probs
                )
                .sum(dim=-1)
                .item()
            )


            vocab_size = (
                probs.shape[-1]
            )

            normalized_entropy = (
                entropy
                / math.log(
                    vocab_size
                )
            )


            top_values, top_indices = (
                torch.topk(
                    probs,
                    k=5,
                    dim=-1,
                )
            )


            top_values = (
                top_values[0]
                .detach()
                .cpu()
                .tolist()
            )

            top_indices = (
                top_indices[0]
                .detach()
                .cpu()
                .tolist()
            )


            top1_prob = float(
                top_values[0]
            )

            top2_prob = float(
                top_values[1]
            )

            top1_top2_margin = (
                top1_prob
                - top2_prob
            )


            top5_mass = float(
                sum(
                    top_values
                )
            )


            hidden_rms = float(
                torch.sqrt(
                    torch.mean(
                        last_hidden
                        ** 2
                    )
                ).item()
            )


            hidden_norm = float(
                torch.linalg
                .vector_norm(
                    last_hidden,
                    dim=-1,
                )
                .item()
            )


            logit_std = float(
                logits.std().item()
            )


            feature_row = {
                "index":
                    index,

                "id":
                    result.get("id"),

                "context_tokens":
                    context_tokens,

                "question_tokens":
                    question_tokens,

                "source_next_token_entropy":
                    entropy,

                "source_next_token_normalized_entropy":
                    normalized_entropy,

                "source_top1_prob":
                    top1_prob,

                "source_top2_prob":
                    top2_prob,

                "source_top1_top2_margin":
                    top1_top2_margin,

                "source_top5_mass":
                    top5_mass,

                "source_last_hidden_rms":
                    hidden_rms,

                "source_last_hidden_norm":
                    hidden_norm,

                "source_logit_std":
                    logit_std,

                # Already observed while building
                # the source cache in the previous run.
                "source_prefill_seconds":
                    float(
                        result[
                            "timing_seconds"
                        ][
                            "source_prefill_existing_work"
                        ]
                    ),

                "top5_token_ids":
                    top_indices,
            }


            out.write(
                json.dumps(
                    feature_row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            out.flush()


            print(
                f"[{number:3d}/"
                f"{len(results):3d}] "
                f"index={index:3d} "
                f"tokens={context_tokens:5d} "
                f"H={entropy:.4f} "
                f"p1={top1_prob:.4f} "
                f"margin={top1_top2_margin:.4f}"
            )


            del input_ids
            del attention_mask
            del backbone_output
            del last_hidden
            del logits
            del log_probs
            del probs

            gc.collect()
            torch.cuda.empty_cache()


    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
