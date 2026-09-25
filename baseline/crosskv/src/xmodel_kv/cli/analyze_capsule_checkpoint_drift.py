from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Localize structural drift between compatible soft-tail checkpoints."
    )
    parser.add_argument("--parent", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2103)
    args = parser.parse_args()
    if args.probes < 1:
        parser.error("probe count must be positive")

    result = analyze_checkpoints(
        load_checkpoint(args.parent),
        load_checkpoint(args.candidate),
        probes=args.probes,
        seed=args.seed,
    )
    result.update(
        {
            "analysis": "soft_tail_checkpoint_drift_localization",
            "parent": args.parent,
            "candidate": args.candidate,
            "probes": args.probes,
            "seed": args.seed,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def load_checkpoint(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("capsule checkpoint must contain a mapping")
    return payload


def analyze_checkpoints(
    parent: dict,
    candidate: dict,
    *,
    probes: int,
    seed: int,
) -> dict:
    parent_embeddings, parent_writer, writer_config = validate_payload(parent)
    candidate_embeddings, candidate_writer, candidate_config = validate_payload(
        candidate
    )
    if parent_embeddings.shape != candidate_embeddings.shape:
        raise ValueError("checkpoint embedding shapes differ")
    if writer_config != candidate_config:
        raise ValueError("checkpoint writer configs differ")
    if set(parent_writer) != set(candidate_writer):
        raise ValueError("checkpoint writer tensor names differ")
    for name in parent_writer:
        if parent_writer[name].shape != candidate_writer[name].shape:
            raise ValueError(f"writer tensor shape differs for {name}")

    tensor_pairs = {"embeddings": (parent_embeddings, candidate_embeddings)}
    tensor_pairs.update(
        {
            f"write_adapter.{name}": (parent_writer[name], candidate_writer[name])
            for name in sorted(parent_writer)
        }
    )
    tensor_metrics = {
        name: tensor_comparison(left, right)
        for name, (left, right) in tensor_pairs.items()
    }
    total_parameter_drift_energy = sum(
        metric["delta_squared_l2"] for metric in tensor_metrics.values()
    )
    for metric in tensor_metrics.values():
        metric["parameter_drift_share"] = safe_ratio(
            metric["delta_squared_l2"], total_parameter_drift_energy
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    hidden_size = int(writer_config["hidden_size"])
    inputs = torch.randn((probes, hidden_size), generator=generator)
    inputs = F.rms_norm(inputs, (hidden_size,))
    num_blocks = int(writer_config["num_hidden_layers"]) - 1
    layer_metrics = []
    with torch.no_grad():
        for layer_index in range(num_blocks):
            scale = float(writer_config["scale"])
            parent_output = block_residual(
                parent_writer, layer_index, inputs, scale=scale
            )
            candidate_output = block_residual(
                candidate_writer, layer_index, inputs, scale=scale
            )
            delta = candidate_output - parent_output
            parent_rms = rms(parent_output)
            candidate_rms = rms(candidate_output)
            delta_rms = rms(delta)
            layer_metrics.append(
                {
                    "layer_index": layer_index,
                    "parent_residual_rms": parent_rms,
                    "candidate_residual_rms": candidate_rms,
                    "drift_rms": delta_rms,
                    "relative_drift_to_parent": safe_ratio(delta_rms, parent_rms),
                    "output_cosine": cosine(parent_output, candidate_output),
                    "functional_drift_energy": float(delta.float().square().sum()),
                }
            )
    total_functional_energy = sum(
        row["functional_drift_energy"] for row in layer_metrics
    )
    for row in layer_metrics:
        row["functional_drift_share"] = safe_ratio(
            row["functional_drift_energy"], total_functional_energy
        )

    ranked = sorted(
        layer_metrics, key=lambda row: row["functional_drift_energy"], reverse=True
    )
    largest_eight = ranked[: min(8, len(ranked))]
    top_eight_share = sum(row["functional_drift_share"] for row in largest_eight)
    window_size = min(8, len(layer_metrics))
    windows = [
        {
            "start": start,
            "end": start + window_size - 1,
            "share": sum(
                row["functional_drift_share"]
                for row in layer_metrics[start : start + window_size]
            ),
        }
        for start in range(len(layer_metrics) - window_size + 1)
    ]
    best_window = max(windows, key=lambda row: row["share"])
    thirds = {
        "early_0_9": share_range(layer_metrics, 0, 9),
        "middle_10_20": share_range(layer_metrics, 10, 20),
        "deep_21_30": share_range(layer_metrics, 21, 30),
    }
    embedding_share = tensor_metrics["embeddings"]["parameter_drift_share"]
    embedding_dominant = embedding_share >= 0.50
    layer_localized = top_eight_share >= 0.60 or best_window["share"] >= 0.60
    if embedding_dominant:
        classification = "embedding_dominant"
    elif layer_localized:
        classification = "layer_localized"
    else:
        classification = "distributed"

    return {
        "architecture": {
            "slots": int(parent_embeddings.shape[0]),
            "hidden_size": hidden_size,
            "writer_blocks": num_blocks,
            "writer_rank": int(writer_config["rank"]),
        },
        "parameter_drift": {
            "total_squared_l2": total_parameter_drift_energy,
            "embedding_share": embedding_share,
            "writer_share": 1.0 - embedding_share,
            "tensors": tensor_metrics,
        },
        "functional_drift": {
            "total_energy": total_functional_energy,
            "third_shares": thirds,
            "largest_eight_layers": [
                int(row["layer_index"]) for row in largest_eight
            ],
            "largest_eight_share": top_eight_share,
            "best_contiguous_eight": best_window,
            "layers": layer_metrics,
        },
        "frozen_thresholds": {
            "embedding_dominant_min_parameter_share": 0.50,
            "layer_localized_min_top8_or_contiguous8_share": 0.60,
        },
        "classification": classification,
    }


def validate_payload(payload: dict) -> tuple[torch.Tensor, dict, dict]:
    embeddings = payload.get("embeddings")
    writer = payload.get("write_adapter_state")
    config = payload.get("write_adapter_config")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("checkpoint embeddings are invalid")
    if not isinstance(writer, dict) or not writer:
        raise ValueError("checkpoint writer state is invalid")
    if not isinstance(config, dict):
        raise ValueError("checkpoint writer config is invalid")
    required = {"hidden_size", "num_hidden_layers", "rank", "scale"}
    if set(config) != required:
        raise ValueError("checkpoint writer config fields are invalid")
    expected_blocks = int(config["num_hidden_layers"]) - 1
    expected_names = {
        f"blocks.{index}.{branch}.weight"
        for index in range(expected_blocks)
        for branch in ("down", "up")
    }
    if set(writer) != expected_names:
        raise ValueError("checkpoint writer tensor structure is invalid")
    return embeddings.float(), {name: value.float() for name, value in writer.items()}, config


def block_residual(
    state: dict, layer_index: int, inputs: torch.Tensor, *, scale: float
) -> torch.Tensor:
    down = state[f"blocks.{layer_index}.down.weight"]
    up = state[f"blocks.{layer_index}.up.weight"]
    return F.linear(F.silu(F.linear(inputs, down)), up) * scale


def tensor_comparison(parent: torch.Tensor, candidate: torch.Tensor) -> dict:
    parent = parent.float()
    candidate = candidate.float()
    delta = candidate - parent
    parent_l2 = float(torch.linalg.vector_norm(parent))
    candidate_l2 = float(torch.linalg.vector_norm(candidate))
    delta_l2 = float(torch.linalg.vector_norm(delta))
    return {
        "elements": parent.numel(),
        "parent_l2": parent_l2,
        "candidate_l2": candidate_l2,
        "delta_l2": delta_l2,
        "delta_squared_l2": delta_l2 * delta_l2,
        "relative_delta_to_parent": safe_ratio(delta_l2, parent_l2),
        "cosine": cosine(parent, candidate),
    }


def rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt())


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    if denominator == 0:
        return 1.0 if torch.equal(left, right) else 0.0
    return float(torch.dot(left, right) / denominator)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0 if numerator == 0 else math.inf
    return float(numerator / denominator)


def share_range(rows: list[dict], start: int, end: int) -> float:
    return sum(
        row["functional_drift_share"]
        for row in rows
        if start <= row["layer_index"] <= end
    )


if __name__ == "__main__":
    main()
