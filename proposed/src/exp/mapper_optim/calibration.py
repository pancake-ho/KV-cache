"""Paired calibration trace collection and paper-specific affine fitting."""
from __future__ import annotations

from contextlib import nullcontext
import gc
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from xmodel_kv.store import ActivationMetadata, ActivationStore

from .cache import legacy_cache, unrotate_rows, prefill
from .inputs import ModelIdentity
from .regression import fit_ridge, select_layers


class Trace:
    def __init__(self, root: Path):
        self.store = ActivationStore(root)
        self.root = root
        self.layers = self.store.metadata.num_layers
        self.heads = self.store.metadata.num_kv_heads
        self.width = self.store.metadata.head_dim

    def array(self, kind: str, layer: int) -> np.ndarray:
        return self.store.open(kind, layer)


@torch.inference_mode()
def collect(identity: ModelIdentity, examples: list[dict], root: Path, *,
            role: str, stride: int, device: torch.device, dtype: torch.dtype,
            capture_factory=None, model_context=None) -> Trace:
    length = len(examples[0]["ids"])
    positions = torch.arange(0, length, stride, device=device)
    count = len(examples) * len(positions)
    metadata = ActivationMetadata(role=role, model_path=identity.name, model_type="qwen3",
                                  num_layers=identity.layers, num_kv_heads=identity.heads,
                                  head_dim=identity.width, num_sequences=len(examples),
                                  sequence_length=length, stride=stride, num_observations=count,
                                  dtype="float32")
    metadata.save(root)
    trace = Trace(root)
    writes = [[trace.store.create(kind, layer) for kind in ("k", "v")]
              for layer in range(identity.layers)]
    weights = []
    model = identity.load(device, dtype)
    capture = active = None
    try:
        with (model_context(model) if model_context else nullcontext(model)) as active:
            capture = capture_factory(active) if capture_factory else None
            if capture is not None:
                weights = [np.lib.format.open_memmap(root / f"attention_{layer:03d}.npy", mode="w+",
                                                    dtype="float32", shape=(2, count, identity.heads))
                           for layer in range(identity.layers)]
            for index, example in enumerate(examples):
                ids = torch.tensor([example["ids"]], dtype=torch.long, device=device)
                cache = prefill(active, ids)
                start = index * len(positions)
                rows = slice(start, start + len(positions))
                for layer, (key, value) in enumerate(unrotate_rows(active, cache)):
                    for writer, data in zip(writes[layer], (key, value), strict=True):
                        writer[rows] = data.index_select(0, positions).float().cpu().numpy()
                if capture is not None:
                    for layer, scores in enumerate(capture.weights(legacy_cache(cache), positions)):
                        weights[layer][:, rows, :] = scores
                del cache
    finally:
        if capture is not None:
            capture.close()
        for pair in writes:
            for writer in pair:
                writer.flush()
        for writer in weights:
            writer.flush()
        del capture, active, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return trace


def fit_layers(source: Trace, target: Trace, artifact: Path, *, settings: dict,
               device: torch.device, fit_dtype: torch.dtype, basis_fn=None,
               weight_shrinkage=None) -> list[dict]:
    head_local = settings["variant"] in {"head-local", "attn-repair"}
    n = source.store.metadata.num_observations
    selected = select_layers(source, target, head_local=settings["method"] == "cachebridge",
                             top_k=settings["top_k"],
                             num_rows=n, sampled=settings["probe_rows"], device=device)
    settings["selected_layers"] = selected
    settings["observations"] = n
    statistics = []
    for layer in range(target.layers):
        state = {}
        for kind_index, kind in enumerate(("k", "v")):
            inputs = [source.array(kind, index) for index in selected[kind][layer]]
            outputs = target.array(kind, layer)
            if weight_shrinkage:
                raw = np.load(target.root / f"attention_{layer:03d}.npy", mmap_mode="r")[kind_index]
            for head in range(target.heads if head_local else 1):
                if head_local:
                    x_blocks = [block[:, head] for block in inputs]
                    y_block = outputs[:, head]
                else:
                    x_blocks = [block.reshape(n, -1) for block in inputs]
                    y_block = outputs.reshape(n, -1)
                width, out = len(x_blocks) * x_blocks[0].shape[1], y_block.shape[1]
                if weight_shrinkage:
                    r = raw[:, head]
                    mean, alpha = weight_shrinkage(r, width)
                    statistics.append({"layer": layer, "kind": kind, "head": head, "alpha": alpha})
                def panels():
                    for begin in range(0, n, settings["chunk_rows"]):
                        sl = slice(begin, begin + settings["chunk_rows"])
                        x = torch.tensor(np.concatenate([block[sl] for block in x_blocks], axis=1),
                                         device=device, dtype=fit_dtype)
                        y = torch.tensor(y_block[sl], device=device, dtype=fit_dtype)
                        w_rows = (1 + alpha * (r[sl] / mean - 1)) if weight_shrinkage else np.ones(len(x))
                        w = torch.tensor(w_rows, device=device, dtype=fit_dtype).reshape(-1, 1)
                        yield x, y, w
                w, b, _, my, gram = fit_ridge(panels, (width, out), ridge=settings["ridge"],
                                              device=device, dtype=fit_dtype)
                prefix = f"{kind}_{head}"
                state[prefix + "_weight"] = w.float().cpu().contiguous()
                state[prefix + "_bias"] = b.float().cpu().contiguous()
                if basis_fn:
                    basis = basis_fn(w, gram, settings["rank"])
                    state[prefix + "_basis"] = basis.float().cpu().contiguous()
                    state[prefix + "_mean"] = my.float().cpu().contiguous()
                    del basis
                del w, b, gram, my
        if head_local:
            state = {f"{kind}_{field}": torch.stack(
                [state[f"{kind}_{head}_{field}"] for head in range(target.heads)])
                for kind in ("k", "v") for field in ("weight", "bias")}
        save_file(state, str(artifact / f"layer_{layer:03d}.safetensors"))
        print(f"Fitted target layer {layer + 1}/{target.layers}", flush=True)
    (artifact / "calibration.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return statistics
