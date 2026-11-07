from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from xmodel_kv.receiver_lens import ReceiverLens
from xmodel_kv.receiver_lens_bank import ReceiverLensBank

from .analyze_receiver_lens import (
    METRICS,
    arm_means,
    compare_arms,
    load_rows,
)


RECEIVER_ID = "longbench_e_paired_lookup_v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply frozen receiver-identity Lens4 development gates."
    )
    parser.add_argument("--identity-correct", required=True)
    parser.add_argument("--identity-shift", required=True)
    parser.add_argument("--shared-correct", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--identity-lens-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2116)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("bootstrap count must be positive")
    groups = {
        "identity_correct": load_rows(args.identity_correct),
        "identity_shift": load_rows(args.identity_shift),
        "shared_correct": load_rows(args.shared_correct),
    }
    integrity = verify_identity_bank(
        base_checkpoint=args.base_checkpoint,
        lens_checkpoint=args.identity_lens_checkpoint,
    )
    result = analyze_identity_groups(
        groups,
        integrity=integrity,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    result.update(
        {
            "analysis": "receiver_identity_lens4_frozen_gates",
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def analyze_identity_groups(
    groups: dict,
    *,
    integrity: dict,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    correct = groups["identity_correct"]
    shift = groups["identity_shift"]
    shared = groups["shared_correct"]
    comparisons = {
        "identity_minus_base": compare_arms(
            correct,
            "lens4",
            correct,
            "base",
            rng=rng,
            replicates=bootstrap_replicates,
        ),
        "identity_minus_hard4": compare_arms(
            correct,
            "lens4",
            correct,
            "hard4",
            rng=rng,
            replicates=bootstrap_replicates,
        ),
        "identity_minus_shared4": compare_arms(
            correct,
            "lens4",
            shared,
            "lens4",
            rng=rng,
            replicates=bootstrap_replicates,
        ),
        "identity_correct_minus_shift": compare_arms(
            correct,
            "lens4",
            shift,
            "lens4",
            rng=rng,
            replicates=bootstrap_replicates,
        ),
    }
    base = comparisons["identity_minus_base"]
    hard = comparisons["identity_minus_hard4"]
    shared_comparison = comparisons["identity_minus_shared4"]
    causal = comparisons["identity_correct_minus_shift"]
    gates = {
        "identity_final_vs_base": (
            base["student_f1"]["mean_delta"] >= 0.06
            and base["student_f1"]["bootstrap_95ci"][0] > 0
        ),
        "identity_final_vs_shared": (
            shared_comparison["student_f1"]["mean_delta"] >= 0.03
            and shared_comparison["student_f1"]["bootstrap_95ci"][0] > 0
        ),
        "identity_bridge_safety": (
            base["bridge_f1"]["mean_delta"] >= -0.03
            and base["bridge_f1"]["bootstrap_95ci"][0] > -0.08
        ),
        "identity_source_causality": all(
            causal[metric]["bootstrap_95ci"][0] > 0 for metric in METRICS
        ),
        "hard4_control": all(
            hard[metric]["mean_delta"] >= 0
            and hard[metric]["bootstrap_95ci"][0] > -0.08
            for metric in METRICS
        ),
        "bank_and_base_integrity": all(integrity["checks"].values()),
    }
    authorized = all(gates.values())
    return {
        "cases": {name: len(rows) for name, rows in groups.items()},
        "means": {name: arm_means(rows) for name, rows in groups.items()},
        "comparisons": comparisons,
        "integrity": integrity,
        "gates": gates,
        "untouched_confirmation_authorized": authorized,
        "conditional_writer_authorized": False,
        "eviction_test_authorized": False,
    }


def verify_identity_bank(*, base_checkpoint: str, lens_checkpoint: str) -> dict:
    base_file_hash = hashlib.sha256(Path(base_checkpoint).read_bytes()).hexdigest()
    payload = torch.load(lens_checkpoint, map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("identity lens checkpoint lacks embeddings")
    lens = ReceiverLens(embeddings)
    bank = ReceiverLensBank({RECEIVER_ID: lens})
    named_parameters = [name for name, _ in bank.named_parameters()]
    checks = {
        "base_file_matches_training": config.get("base_checkpoint_sha256")
        == base_file_hash,
        "base_parameter_hash_unchanged": (
            config.get("base_parameters_unchanged") is True
            and config.get("base_parameter_sha256_before")
            == config.get("base_parameter_sha256_after")
        ),
        "registered_identity_selects_exact_lens": bank.select(RECEIVER_ID) is lens,
        "unknown_identity_bypasses_lens": bank.select("unregistered_agent") is None,
        "bank_has_only_lens_parameters": named_parameters
        == [f"lenses.{RECEIVER_ID}.embeddings"],
        "lens_shape_is_frozen": tuple(embeddings.shape) == (4, 4096),
    }
    return {
        "receiver_id": RECEIVER_ID,
        "base_checkpoint_sha256": base_file_hash,
        "lens_checkpoint_sha256": hashlib.sha256(
            Path(lens_checkpoint).read_bytes()
        ).hexdigest(),
        "named_parameters": named_parameters,
        "checks": checks,
    }


if __name__ == "__main__":
    main()
