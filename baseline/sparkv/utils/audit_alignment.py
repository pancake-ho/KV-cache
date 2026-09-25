from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from baseline.sparkv.utils.plot_results import load_result_records


PAPER_URL = "https://arxiv.org/abs/2604.21231"


@dataclass(frozen=True)
class Check:
    area: str
    status: str
    evidence: str
    implication: str


def _all(rows: list[dict[str, Any]], predicate) -> bool:
    return bool(rows) and all(predicate(row) for row in rows)


def _median_ratio(
    rows: list[dict[str, Any]],
    numerator: str,
    denominator: str,
) -> float:
    values = []
    for row in rows:
        try:
            num = float(row[numerator])
            den = float(row[denominator])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(num) and math.isfinite(den) and den > 0:
            values.append(num / den)
    return float(np.median(values)) if values else float("nan")


def _predictor_checks(path: Path | None) -> list[Check]:
    if path is None:
        return [
            Check(
                "Computation-latency predictor artifact",
                "not_verifiable",
                "No predictor checkpoint was supplied to the audit.",
                "The 3-48-24-1 / 6000 / 80:20 / SGD-MSE contract is not verified.",
            )
        ]
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("train_config", {})
    features = payload.get("feature_names", [])
    state = payload.get("state_dict", {})
    shapes = {key: tuple(value.shape) for key, value in state.items()}
    expected_shapes = {
        "net.0.weight": (48, 3),
        "net.2.weight": (24, 48),
        "net.4.weight": (1, 24),
    }
    architecture_ok = all(shapes.get(key) == value for key, value in expected_shapes.items())
    protocol_ok = (
        int(config.get("samples", -1)) == 6000
        and math.isclose(float(config.get("train_fraction", -1)), 0.8)
        and features == ["token_block", "active_blocks", "gpu_util"]
    )
    metrics = payload.get("metrics", {})
    test_mape = float(metrics.get("test_mape", float("nan")))
    status = "matched" if architecture_ok and protocol_ok and math.isfinite(test_mape) else "partial"
    return [
        Check(
            "Computation-latency predictor",
            status,
            (
                f"architecture={architecture_ok}, protocol={protocol_ok}, "
                f"held-out MAPE={test_mape:.6g}"
            ),
            (
                "The paper discloses the architecture/training split but not optimizer "
                "hyperparameters or an absolute MAPE acceptance threshold."
            ),
        )
    ]


def audit(
    rows: list[dict[str, Any]],
    predictor: Path | None = None,
) -> dict[str, Any]:
    sparkv = [row for row in rows if row.get("strategy", "sparkv") == "sparkv"]
    strategies = sorted({str(row.get("strategy", "sparkv")) for row in rows})
    checks: list[Check] = []

    checks.append(
        Check(
            "KV unit indexing",
            "matched"
            if _all(
                sparkv,
                lambda row: "(token,layer,kv-head)"
                in str(row.get("stream_granularity", "")),
            )
            else "not_verifiable",
            "Runtime records report (token, layer, KV-head) ownership."
            if sparkv
            else "No SparKV runtime rows.",
            "The paper uses c=(t,l,h) with 1024-token chunks; chunk size must be checked in schedules.",
        )
    )
    checks.append(
        Check(
            "Quantization and Huffman bitstream",
            "partial"
            if _all(sparkv, lambda row: bool(row.get("actual_huffman_bitstream")))
            else "missing",
            "Actual Huffman bitstreams are reported by every SparKV run."
            if sparkv
            else "No SparKV runtime rows.",
            "Layer-wise non-uniform bit allocation is not author-disclosed and cannot be claimed exact.",
        )
    )
    checks.append(
        Check(
            "SpargeAttention execution",
            "matched"
            if _all(
                sparkv,
                lambda row: "official Ampere CUDA binding"
                in str(row.get("spargeattention", "")),
            )
            else "missing",
            "Every SparKV run identifies the official Ampere CUDA binding.",
            "This verifies the runtime path, not cross-model portability.",
        )
    )
    checks.append(
        Check(
            "Runtime controller",
            "matched"
            if _all(sparkv, lambda row: bool(row.get("runtime_adaptation")))
            else "missing",
            "Sliding-window route migration is active for SparKV rows.",
            "Window, imbalance margin, and migration cap remain disclosed experiment parameters.",
        )
    )

    network_simulated = any(
        "simulated" in str(row.get("measurement_scope", "")).lower()
        or "preloaded cloud bytes" in str(row.get("measurement_scope", "")).lower()
        for row in sparkv
    )
    checks.append(
        Check(
            "Wireless and cloud path",
            "partial" if network_simulated else "not_verifiable",
            "Cloud artifacts are preloaded and bandwidth is charged in software."
            if network_simulated
            else "No explicit real/simulated network declaration in the records.",
            "This is a controlled network model, not the paper's real Aliyun/Wi-Fi testbed.",
        )
    )

    paper_baselines = {"local_sparse", "strong_hybrid"}
    baseline_status = "partial" if paper_baselines.issubset(strategies) else "missing"
    missing = sorted(paper_baselines - set(strategies))
    checks.append(
        Check(
            "Paper evaluation baselines",
            baseline_status,
            f"strategies={strategies}; missing required local/hybrid={missing}",
            "CacheGen remains absent even when Local Prefill and Strong Hybrid are present.",
        )
    )

    sample_count = len({row.get("sample_index") for row in rows})
    repeat_count = len({row.get("repeat", 0) for row in rows})
    checks.append(
        Check(
            "Evaluation coverage",
            "partial" if sample_count >= 4 and repeat_count >= 3 else "missing",
            f"samples={sample_count}, timing repeats={repeat_count}, observed datasets<=1",
            "The paper evaluates five models and nine datasets; this remains a focused prototype run.",
        )
    )

    scheduler_gap = _median_ratio(
        sparkv,
        "actual_context_rebuild_ms",
        "schedule_predicted_makespan_ms",
    )
    wait_share = _median_ratio(
        sparkv,
        "physical_dependency_wait_ms",
        "actual_context_rebuild_ms",
    )
    decode_share = _median_ratio(
        sparkv,
        "decode_ms",
        "actual_context_rebuild_ms",
    )
    physical_status = (
        "matched"
        if math.isfinite(scheduler_gap) and scheduler_gap <= 1.25
        else "partial"
    )
    checks.append(
        Check(
            "Scheduler-to-physical agreement",
            physical_status,
            (
                f"median actual/predicted={scheduler_gap:.4g}, "
                f"dependency-wait share={wait_share:.4g}, decode share={decode_share:.4g}"
            ),
            "Large gaps must be reported as prototype realization overhead, not SparKV speedup.",
        )
    )
    checks.extend(_predictor_checks(predictor))

    status_counts = {
        status: sum(check.status == status for check in checks)
        for status in ["matched", "partial", "missing", "not_verifiable"]
    }
    return {
        "schema": "sparkv-paper-alignment-audit-v1",
        "paper": PAPER_URL,
        "records": len(rows),
        "sparkv_records": len(sparkv),
        "strategies": strategies,
        "status_counts": status_counts,
        "diagnostics": {
            "median_actual_over_predicted": scheduler_gap,
            "median_dependency_wait_share": wait_share,
            "median_decode_share": decode_share,
        },
        "checks": [asdict(check) for check in checks],
        "claim_guardrail": (
            "matched means the disclosed mechanism is evidenced. It does not mean the "
            "prototype reproduces the paper's complete hardware, codec, datasets, or results."
        ),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SparKV paper-alignment audit",
        "",
        f"Paper: {payload['paper']}",
        "",
        f"Records: {payload['records']} (SparKV: {payload['sparkv_records']})",
        "",
        "| Area | Status | Evidence | Implication |",
        "|---|---|---|---|",
    ]
    for check in payload["checks"]:
        cells = [
            str(check[key]).replace("|", "\\|").replace("\n", " ")
            for key in ["area", "status", "evidence", "implication"]
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Claim guardrail",
            "",
            payload["claim_guardrail"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--predictor", default=None)
    parser.add_argument("--json", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()

    rows = load_result_records([Path(name) for name in args.inputs])
    payload = audit(
        rows,
        predictor=Path(args.predictor) if args.predictor else None,
    )
    json_path = Path(args.json)
    markdown_path = Path(args.markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
