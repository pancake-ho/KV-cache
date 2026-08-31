from __future__ import annotations

import argparse
import gc
import heapq
import json
import math
import os
import random
import re
import string
import threading
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
import torch
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DynamicCache,
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


def middle_truncate(ids: list[int], length: int) -> list[int]:
    if len(ids) <= length:
        return ids
    left = length // 2
    return ids[:left] + ids[-(length - left) :]


def prepare(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        dataset = load_dataset("THUDM/LongBench", "triviaqa", split="test")
    except Exception:
        dataset = load_dataset("zai-org/LongBench", "triviaqa", split="test")

    records: list[dict[str, Any]] = []
    for row_idx, row in enumerate(dataset):
        text = PROMPT.format(context=row["context"], input=row["input"])
        ids = tokenizer(text, add_special_tokens=True).input_ids
        if len(ids) < args.prompt_tokens:
            continue
        ids = middle_truncate(ids, args.prompt_tokens)
        answers = row["answers"]
        if isinstance(answers, str):
            answers = [answers]
        records.append(
            {
                "sample_id": str(row.get("_id", row_idx)),
                "prefill_ids": ids[:-1],
                "seed_id": ids[-1],
                "answers": list(answers),
            }
        )
        if len(records) == args.samples:
            break

    if len(records) < args.samples:
        raise RuntimeError(
            f"Only {len(records)} samples have at least {args.prompt_tokens} tokens."
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, out)
    print(json.dumps({"saved": str(out), "samples": len(records)}, indent=2))


@dataclass(frozen=True)
class ModelRuntime:
    device: torch.device
    dtype: torch.dtype
    backend: str

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"


def cpu_dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported CPU dtype: {name}")


def device_synchronize(runtime: ModelRuntime) -> None:
    if runtime.is_cuda:
        torch.cuda.synchronize(runtime.device)


def clear_device_cache(runtime: ModelRuntime) -> None:
    if runtime.is_cuda:
        torch.cuda.empty_cache()


def describe_runtime(runtime: ModelRuntime) -> dict[str, str]:
    return {
        "device": str(runtime.device),
        "dtype": str(runtime.dtype).removeprefix("torch."),
        "backend": runtime.backend,
    }


def load_model(
    model_id: str,
    requested_device: str = "auto",
    cpu_dtype_name: str = "float32",
):
    if requested_device not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported device selection: {requested_device}")

    # Tokenizer/network failures are unrelated to CUDA and must not be hidden by
    # the CPU fallback path.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    cuda_error: str | None = None

    if requested_device in {"auto", "cuda"}:
        if torch.cuda.is_available():
            try:
                bf16 = torch.cuda.is_bf16_supported()
                compute_dtype = torch.bfloat16 if bf16 else torch.float16
                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quant,
                    device_map={"": 0},
                    attn_implementation="sdpa",
                    torch_dtype=compute_dtype,
                    low_cpu_mem_usage=True,
                )
                model.eval()
                model.config.use_cache = True
                runtime = ModelRuntime(
                    device=torch.device("cuda:0"),
                    dtype=compute_dtype,
                    backend="cuda-nf4",
                )
                print(json.dumps({"model_runtime": describe_runtime(runtime)}))
                return model, tokenizer, runtime
            except Exception as exc:
                if requested_device == "cuda":
                    raise RuntimeError("Strict CUDA model loading failed.") from exc
                cuda_error = f"{type(exc).__name__}: {exc}"
                warnings.warn(
                    "CUDA model loading failed; retrying on CPU without "
                    f"bitsandbytes quantization. Cause: {cuda_error}",
                    RuntimeWarning,
                )
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        elif requested_device == "cuda":
            raise RuntimeError(
                "--device cuda was requested, but torch.cuda.is_available() is False."
            )

    cpu_dtype = cpu_dtype_from_name(cpu_dtype_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": "cpu"},
            attn_implementation="sdpa",
            torch_dtype=cpu_dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as cpu_error:
        if cuda_error is not None:
            raise RuntimeError(
                "Both CUDA loading and the CPU fallback failed. "
                f"CUDA cause: {cuda_error}; "
                f"CPU cause: {type(cpu_error).__name__}: {cpu_error}"
            ) from cpu_error
        raise

    model.eval()
    model.config.use_cache = True
    runtime = ModelRuntime(
        device=torch.device("cpu"),
        dtype=cpu_dtype,
        backend=f"cpu-{cpu_dtype_name}",
    )
    print(json.dumps({"model_runtime": describe_runtime(runtime)}))
    return model, tokenizer, runtime


def to_legacy(cache: Any):
    return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache


def quantize_uniform_5bit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Symmetric, per-KV-head uniform quantization. q uses [0, 31].
    x = x.float()
    scale = x.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8) / 15.0
    q = (torch.round(x / scale).clamp(-16, 15) + 16).to(torch.uint8)
    return q.contiguous(), scale.contiguous()


def huffman_payload_bytes(q: torch.Tensor) -> int:
    # Exact optimal Huffman payload length from the 32-symbol histogram.
    counts = np.bincount(q.reshape(-1).numpy(), minlength=32)
    heap = [int(x) for x in counts if x > 0]
    if len(heap) <= 1:
        payload_bits = int(sum(heap))
    else:
        heapq.heapify(heap)
        payload_bits = 0
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            payload_bits += a + b
            heapq.heappush(heap, a + b)
    # Store a 32-entry frequency table so the decoder can reconstruct the tree.
    return math.ceil(payload_bits / 8) + 32 * 4


def build_cache(args: argparse.Namespace) -> None:
    records = torch.load(args.prepared, map_location="cpu", weights_only=False)
    model, _, runtime = load_model(args.model, args.device, args.cpu_dtype)
    root = Path(args.cache_root)
    formats = set(args.formats)

    for sample_idx, record in enumerate(records[: args.samples]):
        ids = torch.tensor(
            record["prefill_ids"], dtype=torch.long, device=runtime.device
        )[None]
        if ids.shape[1] % args.chunk_size != 0:
            raise ValueError("prefill length must be divisible by chunk_size")
        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
        legacy = to_legacy(output.past_key_values)
        num_chunks = ids.shape[1] // args.chunk_size
        layer_count = len(legacy)
        head_count = legacy[0][0].shape[1]

        sample_name = f"sample_{sample_idx:03d}"
        roots = {fmt: root / fmt / sample_name for fmt in formats}
        for directory in roots.values():
            directory.mkdir(parents=True, exist_ok=True)
        meta: dict[str, dict[str, Any]] = {
            fmt: {
                "seq_len": ids.shape[1],
                "chunk_size": args.chunk_size,
                "num_chunks": num_chunks,
                "layers": layer_count,
                "kv_heads": head_count,
                "runtime": describe_runtime(runtime),
                "chunks": [],
            }
            for fmt in formats
        }

        for t in range(num_chunks):
            start = t * args.chunk_size
            end = start + args.chunk_size
            raw_tensors: dict[str, torch.Tensor] = {}
            q5_tensors: dict[str, torch.Tensor] = {}
            raw_lh: dict[str, int] = {}
            q5_lh: dict[str, int] = {}

            for layer, (key, value) in enumerate(legacy):
                k = key[:, :, start:end, :].detach().cpu().contiguous()
                v = value[:, :, start:end, :].detach().cpu().contiguous()
                raw_tensors[f"k_{layer:02d}"] = k
                raw_tensors[f"v_{layer:02d}"] = v
                for h in range(head_count):
                    raw_lh[f"{layer}:{h}"] = (
                        k[:, h].numel() + v[:, h].numel()
                    ) * k.element_size()

                if "q5" in formats:
                    qk, sk = quantize_uniform_5bit(k)
                    qv, sv = quantize_uniform_5bit(v)
                    q5_tensors[f"qk_{layer:02d}"] = qk
                    q5_tensors[f"qv_{layer:02d}"] = qv
                    q5_tensors[f"sk_{layer:02d}"] = sk
                    q5_tensors[f"sv_{layer:02d}"] = sv
                    for h in range(head_count):
                        q5_lh[f"{layer}:{h}"] = (
                            huffman_payload_bytes(qk[:, h].contiguous())
                            + huffman_payload_bytes(qv[:, h].contiguous())
                            + 2 * 4
                        )

            if "raw" in formats:
                path = roots["raw"] / f"chunk_{t:03d}.safetensors"
                save_file(raw_tensors, str(path))
                meta["raw"]["chunks"].append(
                    {
                        "index": t,
                        "wire_bytes": int(sum(raw_lh.values())),
                        "lh_wire_bytes": raw_lh,
                    }
                )
            if "q5" in formats:
                path = roots["q5"] / f"chunk_{t:03d}.safetensors"
                save_file(q5_tensors, str(path))
                meta["q5"]["chunks"].append(
                    {
                        "index": t,
                        "wire_bytes": int(sum(q5_lh.values())),
                        "lh_wire_bytes": q5_lh,
                    }
                )

        for fmt, directory in roots.items():
            (directory / "meta.json").write_text(
                json.dumps(meta[fmt], indent=2), encoding="utf-8"
            )
        print(f"built {sample_name}: {num_chunks} chunks")
        del ids, output, legacy
        clear_device_cache(runtime)


def load_meta(sample_dir: Path) -> dict[str, Any]:
    return json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))


@dataclass
class FetchStats:
    wire_bytes: int = 0
    disk_ms: float = 0.0
    wire_ms: float = 0.0
    decode_ms: float = 0.0


class MeetingState:
    def __init__(self, chunks: int):
        self.low = 0
        self.high = chunks - 1
        self.lock = threading.Lock()

    def claim_compute(self) -> int | None:
        with self.lock:
            if self.low > self.high:
                return None
            value = self.low
            self.low += 1
            return value

    def claim_fetch(self) -> int | None:
        with self.lock:
            if self.low > self.high:
                return None
            value = self.high
            self.high -= 1
            return value


def effective_bandwidth_mbps(mean_mbps: float, cv: float, rng: np.random.Generator):
    if cv <= 0:
        return mean_mbps
    sigma2 = math.log1p(cv * cv)
    factor = rng.lognormal(mean=-0.5 * sigma2, sigma=math.sqrt(sigma2))
    return max(1e-3, mean_mbps * factor)


def decode_chunk(
    cpu: dict[str, torch.Tensor],
    fmt: str,
    layers: int,
    target_dtype: torch.dtype,
):
    decoded: dict[str, torch.Tensor] = {}
    if fmt == "raw":
        for layer in range(layers):
            decoded[f"k_{layer:02d}"] = cpu[f"k_{layer:02d}"].to(target_dtype)
            decoded[f"v_{layer:02d}"] = cpu[f"v_{layer:02d}"].to(target_dtype)
        return decoded
    for layer in range(layers):
        qk = cpu[f"qk_{layer:02d}"].float() - 16.0
        qv = cpu[f"qv_{layer:02d}"].float() - 16.0
        decoded[f"k_{layer:02d}"] = (
            qk * cpu[f"sk_{layer:02d}"].float()
        ).to(target_dtype)
        decoded[f"v_{layer:02d}"] = (
            qv * cpu[f"sv_{layer:02d}"].float()
        ).to(target_dtype)
    return decoded


def fetch_one_chunk(
    sample_dir: Path,
    meta: dict[str, Any],
    fmt: str,
    t: int,
    bandwidth_mbps: float,
    jitter_cv: float,
    rng: np.random.Generator,
    runtime: ModelRuntime,
    copy_stream: torch.cuda.Stream | None,
) -> tuple[dict[str, torch.Tensor], FetchStats]:
    stats = FetchStats()
    begin = time.perf_counter()
    cpu = load_file(str(sample_dir / f"chunk_{t:03d}.safetensors"), device="cpu")
    stats.disk_ms = (time.perf_counter() - begin) * 1000

    stats.wire_bytes = int(meta["chunks"][t]["wire_bytes"])
    bw = effective_bandwidth_mbps(bandwidth_mbps, jitter_cv, rng)
    delay = stats.wire_bytes * 8 / (bw * 1e6)
    time.sleep(delay)
    stats.wire_ms = delay * 1000

    begin = time.perf_counter()
    decoded = decode_chunk(cpu, fmt, int(meta["layers"]), runtime.dtype)
    stats.decode_ms = (time.perf_counter() - begin) * 1000

    resident: dict[str, torch.Tensor] = {}
    if runtime.is_cuda:
        assert copy_stream is not None
        with torch.cuda.stream(copy_stream):
            for name, tensor in decoded.items():
                try:
                    tensor = tensor.pin_memory()
                except RuntimeError:
                    pass
                resident[name] = tensor.to(
                    runtime.device, non_blocking=True
                )
        return resident, stats

    # CPU path: no pinning, CUDA stream, or asynchronous device copy.
    for name, tensor in decoded.items():
        resident[name] = tensor.to(runtime.device)
    return resident, stats


def append_stats(total: FetchStats, item: FetchStats) -> None:
    total.wire_bytes += item.wire_bytes
    total.disk_ms += item.disk_ms
    total.wire_ms += item.wire_ms
    total.decode_ms += item.decode_ms


def compute_one_chunk(
    model,
    prefill_ids: list[int],
    chunk_size: int,
    t: int,
    past: Any,
    runtime: ModelRuntime,
):
    start = t * chunk_size
    current = torch.tensor(
        prefill_ids[start : start + chunk_size],
        dtype=torch.long,
        device=runtime.device,
    )[None]
    past_len = 0 if past is None else int(past.get_seq_length())
    attention_mask = torch.ones(
        (1, past_len + current.shape[1]),
        dtype=torch.long,
        device=runtime.device,
    )
    cache_position = torch.arange(
        past_len,
        past_len + current.shape[1],
        dtype=torch.long,
        device=runtime.device,
    )
    with torch.inference_mode():
        output = model(
            input_ids=current,
            attention_mask=attention_mask,
            past_key_values=past,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
    return output.past_key_values


def assemble_cache(
    local_cache: Any,
    local_chunks: int,
    fetched: dict[int, dict[str, torch.Tensor]],
    meta: dict[str, Any],
) -> DynamicCache:
    local = None if local_cache is None else to_legacy(local_cache)
    legacy = []
    for layer in range(int(meta["layers"])):
        keys, values = [], []
        if local_chunks:
            keys.append(local[layer][0])
            values.append(local[layer][1])
        for t in sorted(fetched):
            keys.append(fetched[t][f"k_{layer:02d}"])
            values.append(fetched[t][f"v_{layer:02d}"])
        k = torch.cat(keys, dim=-2) if len(keys) > 1 else keys[0]
        v = torch.cat(values, dim=-2) if len(values) > 1 else values[0]
        if k.shape[-2] != int(meta["seq_len"]):
            raise RuntimeError(f"assembled length mismatch at layer {layer}")
        legacy.append((k, v))
    return DynamicCache.from_legacy_cache(tuple(legacy))


def first_step(
    model,
    seed_id: int,
    cache: DynamicCache,
    runtime: ModelRuntime,
):
    past_len = int(cache.get_seq_length())
    current = torch.tensor([[seed_id]], dtype=torch.long, device=runtime.device)
    mask = torch.ones(
        (1, past_len + 1), dtype=torch.long, device=runtime.device
    )
    position = torch.tensor(
        [past_len], dtype=torch.long, device=runtime.device
    )
    with torch.inference_mode():
        output = model(
            input_ids=current,
            attention_mask=mask,
            past_key_values=cache,
            cache_position=position,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
    token = int(output.logits[:, -1].argmax(dim=-1).item())
    return token, output.past_key_values


def continue_greedy(
    model,
    first_token: int,
    cache: Any,
    max_new_tokens: int,
    eos: int,
    runtime: ModelRuntime,
):
    generated = [first_token]
    current = first_token
    while len(generated) < max_new_tokens and current != eos:
        past_len = int(cache.get_seq_length())
        ids = torch.tensor(
            [[current]], dtype=torch.long, device=runtime.device
        )
        mask = torch.ones(
            (1, past_len + 1), dtype=torch.long, device=runtime.device
        )
        position = torch.tensor(
            [past_len], dtype=torch.long, device=runtime.device
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
        current = int(output.logits[:, -1].argmax(dim=-1).item())
        generated.append(current)
        cache = output.past_key_values
    return generated


class PowerSampler:
    def __init__(self, period: float = 0.02, enabled: bool = True):
        self.period = period
        self.enabled = enabled
        self.samples: list[tuple[float, float]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.handle = None

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return

        def sample() -> None:
            while not self.stop_event.is_set():
                now = time.perf_counter()
                watts = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                self.samples.append((now, watts))
                self.stop_event.wait(self.period)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()

    def stop(self) -> float | None:
        if self.thread is None:
            return None
        self.stop_event.set()
        self.thread.join()
        if len(self.samples) < 2:
            return None
        return float(
            sum(
                0.5 * (p0 + p1) * (t1 - t0)
                for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:])
            )
        )


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction).split()
    best = 0.0
    for answer in answers:
        gold = normalize_answer(answer).split()
        common = Counter(pred) & Counter(gold)
        overlap = sum(common.values())
        if not pred or not gold:
            score = float(pred == gold)
        elif overlap == 0:
            score = 0.0
        else:
            precision = overlap / len(pred)
            recall = overlap / len(gold)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def run_request(
    model,
    tokenizer,
    runtime: ModelRuntime,
    record: dict[str, Any],
    sample_dir: Path,
    fmt: str,
    strategy: str,
    split: int,
    bandwidth_mbps: float,
    jitter_cv: float,
    quality_tokens: int,
    rng_seed: int,
) -> dict[str, Any]:
    meta = load_meta(sample_dir)
    chunks = int(meta["num_chunks"])
    chunk_size = int(meta["chunk_size"])
    fetched: dict[int, dict[str, torch.Tensor]] = {}
    fetch_total = FetchStats()
    fetch_error: list[BaseException] = []
    local_cache = None
    local_chunks = 0
    copy_stream = torch.cuda.Stream() if runtime.is_cuda else None
    rng = np.random.default_rng(rng_seed)

    state = MeetingState(chunks) if strategy == "adaptive" else None
    if strategy == "local":
        split = chunks
    elif strategy == "fetch":
        split = 0
    elif strategy == "static":
        if not 0 <= split <= chunks:
            raise ValueError("split must be between 0 and num_chunks")
    elif strategy != "adaptive":
        raise ValueError(strategy)

    def fetch_worker() -> None:
        nonlocal fetch_total
        try:
            if runtime.is_cuda:
                torch.cuda.set_device(runtime.device)
            if strategy == "adaptive":
                assert state is not None
                getter: Callable[[], int | None] = state.claim_fetch
            else:
                indices = iter(range(chunks - 1, split - 1, -1))
                getter = lambda: next(indices, None)
            while True:
                t = getter()
                if t is None:
                    break
                resident, stats = fetch_one_chunk(
                    sample_dir,
                    meta,
                    fmt,
                    t,
                    bandwidth_mbps,
                    jitter_cv,
                    rng,
                    runtime,
                    copy_stream,
                )
                fetched[t] = resident
                append_stats(fetch_total, stats)
            if copy_stream is not None:
                copy_stream.synchronize()
        except BaseException as exc:
            fetch_error.append(exc)

    device_synchronize(runtime)
    if runtime.is_cuda:
        torch.cuda.reset_peak_memory_stats(runtime.device)
    power = PowerSampler(enabled=runtime.is_cuda)
    power.start()
    request_start = time.perf_counter()

    thread = None
    if strategy != "local":
        thread = threading.Thread(target=fetch_worker, daemon=True)
        thread.start()

    if strategy == "adaptive":
        assert state is not None
        while True:
            t = state.claim_compute()
            if t is None:
                break
            local_cache = compute_one_chunk(
                model,
                record["prefill_ids"],
                chunk_size,
                t,
                local_cache,
                runtime,
            )
            local_chunks += 1
    else:
        for t in range(split):
            local_cache = compute_one_chunk(
                model,
                record["prefill_ids"],
                chunk_size,
                t,
                local_cache,
                runtime,
            )
            local_chunks += 1

    if thread is not None:
        thread.join()
    if fetch_error:
        raise fetch_error[0]

    context_cache = assemble_cache(local_cache, local_chunks, fetched, meta)
    first, decode_cache = first_step(
        model, int(record["seed_id"]), context_cache, runtime
    )
    device_synchronize(runtime)
    ttft_ms = (time.perf_counter() - request_start) * 1000
    ttft_energy_j = power.stop()

    generated = continue_greedy(
        model,
        first,
        decode_cache,
        max_new_tokens=max(1, quality_tokens),
        eos=tokenizer.eos_token_id,
        runtime=runtime,
    )
    prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {
        "strategy": strategy,
        "format": fmt,
        "ttft_ms": ttft_ms,
        "ttft_energy_j": ttft_energy_j,
        "local_chunks": local_chunks,
        "fetched_chunks": len(fetched),
        "wire_bytes": fetch_total.wire_bytes,
        "disk_ms": fetch_total.disk_ms,
        "wire_ms": fetch_total.wire_ms,
        "decode_ms": fetch_total.decode_ms,
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated(runtime.device) / 2**20
            if runtime.is_cuda
            else 0.0
        ),
        "rss_mib": psutil.Process().memory_info().rss / 2**20,
        "runtime_device": str(runtime.device),
        "runtime_dtype": str(runtime.dtype).removeprefix("torch."),
        "runtime_backend": runtime.backend,
        "prediction": prediction,
        "f1": qa_f1(prediction, record["answers"]),
    }


def run(args: argparse.Namespace) -> None:
    records = torch.load(args.prepared, map_location="cpu", weights_only=False)
    model, tokenizer, runtime = load_model(
        args.model, args.device, args.cpu_dtype
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Kernel/model warm-up; this is intentionally outside TTFT.
    warm = torch.tensor(
        [[1, 2, 3, 4]], dtype=torch.long, device=runtime.device
    )
    with torch.inference_mode():
        model(input_ids=warm, use_cache=True, logits_to_keep=1)
    device_synchronize(runtime)

    with output.open("w", encoding="utf-8") as handle:
        for sample_idx, record in enumerate(records[: args.samples]):
            sample_dir = Path(args.cache_root) / args.format / f"sample_{sample_idx:03d}"
            for strategy in args.strategies:
                for repeat in range(args.repeats):
                    result = run_request(
                        model=model,
                        tokenizer=tokenizer,
                        runtime=runtime,
                        record=record,
                        sample_dir=sample_dir,
                        fmt=args.format,
                        strategy=strategy,
                        split=args.split,
                        bandwidth_mbps=args.bandwidth_mbps,
                        jitter_cv=args.jitter_cv,
                        quality_tokens=args.quality_tokens if repeat == 0 else 1,
                        rng_seed=args.seed + sample_idx * 1000 + repeat,
                    )
                    result.update(
                        {
                            "sample_index": sample_idx,
                            "sample_id": record["sample_id"],
                            "repeat": repeat,
                            "bandwidth_mbps": args.bandwidth_mbps,
                            "jitter_cv": args.jitter_cv,
                        }
                    )
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(json.dumps(result, ensure_ascii=False))
                    clear_device_cache(runtime)


def profile(args: argparse.Namespace) -> None:
    records = torch.load(args.prepared, map_location="cpu", weights_only=False)
    model, _, runtime = load_model(args.model, args.device, args.cpu_dtype)
    layers = list(model.model.layers)
    handles = []

    if runtime.is_cuda:
        starts = [torch.cuda.Event(enable_timing=True) for _ in layers]
        ends = [torch.cuda.Event(enable_timing=True) for _ in layers]
        for layer_idx, layer in enumerate(layers):
            handles.append(
                layer.register_forward_pre_hook(
                    lambda _m, _a, idx=layer_idx: starts[idx].record()
                )
            )
            handles.append(
                layer.register_forward_hook(
                    lambda _m, _a, _o, idx=layer_idx: ends[idx].record()
                )
            )
    else:
        cpu_starts = [0.0 for _ in layers]
        cpu_elapsed_ms = [0.0 for _ in layers]

        def make_cpu_pre_hook(index: int):
            def hook(_module, _args):
                cpu_starts[index] = time.perf_counter()

            return hook

        def make_cpu_post_hook(index: int):
            def hook(_module, _args, _output):
                cpu_elapsed_ms[index] = (
                    time.perf_counter() - cpu_starts[index]
                ) * 1000.0

            return hook

        for layer_idx, layer in enumerate(layers):
            handles.append(
                layer.register_forward_pre_hook(make_cpu_pre_hook(layer_idx))
            )
            handles.append(
                layer.register_forward_hook(make_cpu_post_hook(layer_idx))
            )

    all_samples = []
    try:
        for record in records[: args.samples]:
            past = None
            sample = []
            chunks = len(record["prefill_ids"]) // args.chunk_size
            for t in range(chunks):
                past = compute_one_chunk(
                    model,
                    record["prefill_ids"],
                    args.chunk_size,
                    t,
                    past,
                    runtime,
                )
                device_synchronize(runtime)
                if runtime.is_cuda:
                    sample.append(
                        [
                            starts[layer].elapsed_time(ends[layer])
                            for layer in range(len(layers))
                        ]
                    )
                else:
                    sample.append(list(cpu_elapsed_ms))
            all_samples.append(sample)
            del past
            clear_device_cache(runtime)
    finally:
        for handle in handles:
            handle.remove()

    costs = np.asarray(all_samples, dtype=np.float64)
    result = {
        "model": args.model,
        "chunk_size": args.chunk_size,
        "samples": len(all_samples),
        "runtime": describe_runtime(runtime),
        "token_layer_ms_median": np.median(costs, axis=0).tolist(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"saved": args.output, "shape": list(costs.shape)}, indent=2))


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help=(
            "auto: try CUDA NF4 and fall back to CPU; "
            "cuda: require CUDA; cpu: force CPU"
        ),
    )
    parser.add_argument(
        "--cpu-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="dtype used only by the CPU model path",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--prompt-tokens", type=int, default=8193)
    p.add_argument("--output", required=True)
    p.set_defaults(func=prepare)

    p = sub.add_parser("build-cache")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--prepared", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--formats", nargs="+", choices=["raw", "q5"], default=["raw", "q5"])
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--chunk-size", type=int, default=1024)
    add_runtime_arguments(p)
    p.set_defaults(func=build_cache)

    p = sub.add_parser("profile")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--prepared", required=True)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--output", required=True)
    add_runtime_arguments(p)
    p.set_defaults(func=profile)

    p = sub.add_parser("run")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--prepared", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--format", choices=["raw", "q5"], default="raw")
    p.add_argument(
        "--strategies",
        nargs="+",
        choices=["local", "fetch", "static", "adaptive"],
        default=["local", "fetch", "static", "adaptive"],
    )
    p.add_argument("--split", type=int, default=4)
    p.add_argument("--bandwidth-mbps", type=float, default=640.0)
    p.add_argument("--jitter-cv", type=float, default=0.0)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--quality-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", required=True)
    add_runtime_arguments(p)
    p.set_defaults(func=run)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    seed_everything(getattr(args, "seed", 2026))
    args.func(args)
