from __future__ import annotations

import gc
import json
import random
import re
import string
import threading
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_ID = "Qwen/Qwen3-4B"

PROMPT = (
    "Answer the question based on the given passage. Only give me the answer "
    "and do not output any other words. The following are some examples.\n\n"
    "{context}\n\n{input}"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def middle_truncate(
    ids: list[int],
    length: int,
) -> list[int]:
    if len(ids) <= length:
        return ids

    left = length // 2
    return (
        ids[:left]
        + ids[-(length - left) :]
    )


def prepare_command(
    args: Any,
) -> None:
    """
    Prepare LongBench/TriviaQA records.

    PROMPT_TOKENS includes the final seed token.  Therefore `prefill_ids`
    contains PROMPT_TOKENS - 1 tokens.  With the default
    PROMPT_TOKENS=8193 and CHUNK_SIZE=1024, the prefill length is exactly
    8192 = 8 * 1024.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    try:
        dataset = load_dataset(
            "THUDM/LongBench",
            "triviaqa",
            split="test",
            trust_remote_code=True,
        )
    except Exception as exc:
        warnings.warn(
            "THUDM/LongBench load failed; retrying with "
            "zai-org/LongBench. "
            f"Cause: {type(exc).__name__}: {exc}",
            RuntimeWarning,
        )
        dataset = load_dataset(
            "zai-org/LongBench",
            "triviaqa",
            split="test",
            trust_remote_code=True,
        )

    records: list[dict[str, Any]] = []

    for row_idx, row in enumerate(dataset):
        text = PROMPT.format(
            context=row["context"],
            input=row["input"],
        )
        ids = tokenizer(
            text,
            add_special_tokens=True,
        ).input_ids

        if len(ids) < args.prompt_tokens:
            continue

        ids = middle_truncate(
            ids,
            args.prompt_tokens,
        )

        answers = row["answers"]
        if isinstance(answers, str):
            answers = [answers]

        records.append(
            {
                "sample_id": str(
                    row.get("_id", row_idx)
                ),
                "prefill_ids": ids[:-1],
                "seed_id": ids[-1],
                "answers": list(answers),
            }
        )

        if len(records) == args.samples:
            break

    if len(records) < args.samples:
        raise RuntimeError(
            "Only "
            f"{len(records)} samples have at least "
            f"{args.prompt_tokens} tokens."
        )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        records,
        output,
    )

    print(
        json.dumps(
            {
                "saved": str(output),
                "samples": len(records),
                "prompt_tokens": int(
                    args.prompt_tokens
                ),
                "prefill_tokens": int(
                    args.prompt_tokens - 1
                ),
            },
            indent=2,
        )
    )


@dataclass(frozen=True)
class ModelRuntime:
    device: torch.device
    dtype: torch.dtype
    backend: str

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"


def cpu_dtype_from_name(
    name: str,
) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16

    raise ValueError(
        f"Unsupported CPU dtype: {name}"
    )


def device_synchronize(
    runtime: ModelRuntime,
) -> None:
    if runtime.is_cuda:
        torch.cuda.synchronize(
            runtime.device
        )


def clear_device_cache(
    runtime: ModelRuntime,
) -> None:
    if runtime.is_cuda:
        torch.cuda.empty_cache()


def describe_runtime(
    runtime: ModelRuntime,
) -> dict[str, str]:
    return {
        "device": str(runtime.device),
        "dtype": str(
            runtime.dtype
        ).removeprefix("torch."),
        "backend": runtime.backend,
    }


def load_model(
    model_id: str,
    requested_device: str = "auto",
    cpu_dtype_name: str = "float32",
):
    """
    Load the model for the SparKV reproduction.

    `requested_device="cuda"` is strict: CUDA or model-loading failure aborts
    instead of silently contaminating a GPU experiment with CPU results.
    """
    if requested_device not in {
        "auto",
        "cuda",
        "cpu",
    }:
        raise ValueError(
            "Unsupported device selection: "
            f"{requested_device}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )

    cuda_error: str | None = None

    if requested_device in {
        "auto",
        "cuda",
    }:
        if torch.cuda.is_available():
            try:
                bf16 = (
                    torch.cuda.is_bf16_supported()
                )
                compute_dtype = (
                    torch.bfloat16
                    if bf16
                    else torch.float16
                )

                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=(
                        compute_dtype
                    ),
                )

                model = (
                    AutoModelForCausalLM
                    .from_pretrained(
                        model_id,
                        quantization_config=quant,
                        device_map={"": 0},
                        attn_implementation="sdpa",
                        torch_dtype=compute_dtype,
                        low_cpu_mem_usage=True,
                    )
                )
                model.eval()
                model.config.use_cache = True

                runtime = ModelRuntime(
                    device=torch.device(
                        "cuda:0"
                    ),
                    dtype=compute_dtype,
                    backend="cuda-nf4",
                )

                print(
                    json.dumps(
                        {
                            "model_runtime":
                                describe_runtime(
                                    runtime
                                )
                        }
                    )
                )

                return (
                    model,
                    tokenizer,
                    runtime,
                )

            except Exception as exc:
                if requested_device == "cuda":
                    raise RuntimeError(
                        "Strict CUDA model loading failed."
                    ) from exc

                cuda_error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
                warnings.warn(
                    "CUDA model loading failed; "
                    "retrying on CPU without "
                    "bitsandbytes quantization. "
                    f"Cause: {cuda_error}",
                    RuntimeWarning,
                )

                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        elif requested_device == "cuda":
            raise RuntimeError(
                "--device cuda was requested, "
                "but torch.cuda.is_available() "
                "is False."
            )

    cpu_dtype = cpu_dtype_from_name(
        cpu_dtype_name
    )

    try:
        model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_id,
                device_map={"": "cpu"},
                attn_implementation="sdpa",
                torch_dtype=cpu_dtype,
                low_cpu_mem_usage=True,
            )
        )

    except Exception as cpu_error:
        if cuda_error is not None:
            raise RuntimeError(
                "Both CUDA loading and the CPU "
                "fallback failed. "
                f"CUDA cause: {cuda_error}; "
                "CPU cause: "
                f"{type(cpu_error).__name__}: "
                f"{cpu_error}"
            ) from cpu_error

        raise

    model.eval()
    model.config.use_cache = True

    runtime = ModelRuntime(
        device=torch.device("cpu"),
        dtype=cpu_dtype,
        backend=f"cpu-{cpu_dtype_name}",
    )

    print(
        json.dumps(
            {
                "model_runtime":
                    describe_runtime(
                        runtime
                    )
            }
        )
    )

    return (
        model,
        tokenizer,
        runtime,
    )


def to_legacy(
    cache: Any,
):
    return (
        cache.to_legacy_cache()
        if hasattr(
            cache,
            "to_legacy_cache",
        )
        else cache
    )


def continue_greedy(
    model: Any,
    first_token: int,
    cache: Any,
    max_new_tokens: int,
    eos: int | None,
    runtime: ModelRuntime,
) -> list[int]:
    generated = [
        int(first_token)
    ]
    current = int(first_token)

    while (
        len(generated)
        < max_new_tokens
        and (
            eos is None
            or current != eos
        )
    ):
        past_len = int(
            cache.get_seq_length()
        )

        ids = torch.tensor(
            [[current]],
            dtype=torch.long,
            device=runtime.device,
        )

        mask = torch.ones(
            (
                1,
                past_len + 1,
            ),
            dtype=torch.long,
            device=runtime.device,
        )

        position = torch.tensor(
            [past_len],
            dtype=torch.long,
            device=runtime.device,
        )

        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=cache,
                cache_position=position,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )

        current = int(
            output.logits[
                :,
                -1,
            ]
            .argmax(
                dim=-1
            )
            .item()
        )
        generated.append(
            current
        )
        cache = (
            output.past_key_values
        )

    return generated


class PowerSampler:
    def __init__(
        self,
        period: float = 0.02,
        enabled: bool = True,
    ) -> None:
        self.period = float(period)
        self.enabled = bool(enabled)

        self.samples: list[
            tuple[
                float,
                float,
            ]
        ] = []

        self.stop_event = (
            threading.Event()
        )
        self.thread: (
            threading.Thread
            | None
        ) = None
        self.handle = None
        self.pynvml = None

    def start(self) -> None:
        if not self.enabled:
            return

        try:
            import pynvml

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = (
                pynvml
                .nvmlDeviceGetHandleByIndex(
                    0
                )
            )

        except Exception:
            return

        def sample() -> None:
            assert (
                self.pynvml
                is not None
            )
            while not (
                self.stop_event
                .is_set()
            ):
                now = (
                    time.perf_counter()
                )
                watts = (
                    self.pynvml
                    .nvmlDeviceGetPowerUsage(
                        self.handle
                    )
                    / 1000.0
                )

                self.samples.append(
                    (
                        now,
                        watts,
                    )
                )
                self.stop_event.wait(
                    self.period
                )

        self.thread = (
            threading.Thread(
                target=sample,
                daemon=True,
                name="sparkv-power",
            )
        )
        self.thread.start()

    def stop(
        self,
    ) -> float | None:
        if self.thread is None:
            return None

        self.stop_event.set()
        self.thread.join()

        if len(
            self.samples
        ) < 2:
            return None

        return float(
            sum(
                0.5
                * (
                    p0
                    + p1
                )
                * (
                    t1
                    - t0
                )
                for (
                    t0,
                    p0,
                ), (
                    t1,
                    p1,
                ) in zip(
                    self.samples,
                    self.samples[
                        1:
                    ],
                )
            )
        )


def normalize_answer(
    text: str,
) -> str:
    text = text.lower()
    text = "".join(
        ch
        for ch in text
        if ch
        not in set(
            string.punctuation
        )
    )
    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )
    return " ".join(
        text.split()
    )


def qa_f1(
    prediction: str,
    answers: list[str],
) -> float:
    pred = (
        normalize_answer(
            prediction
        )
        .split()
    )

    best = 0.0

    for answer in answers:
        gold = (
            normalize_answer(
                answer
            )
            .split()
        )

        common = (
            Counter(pred)
            & Counter(gold)
        )
        overlap = sum(
            common.values()
        )

        if not pred or not gold:
            score = float(
                pred == gold
            )

        elif overlap == 0:
            score = 0.0

        else:
            precision = (
                overlap
                / len(pred)
            )
            recall = (
                overlap
                / len(gold)
            )
            score = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        best = max(
            best,
            score,
        )

    return best
