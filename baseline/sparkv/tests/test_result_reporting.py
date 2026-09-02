import json

from baseline.sparkv.utils.audit_alignment import audit
from baseline.sparkv.utils.plot_results import generate_plots, load_result_records


def _row(sample, strategy="sparkv"):
    return {
        "record_type": "sparkv-direct",
        "strategy": strategy,
        "sample_index": sample,
        "repeat": 0,
        "ttft_ms": 2000.0 + sample * 100,
        "ttft_energy_j": 20.0,
        "f1": 0.9,
        "local_units": 4,
        "streamed_units": 6,
        "schedule_predicted_makespan_ms": 1000.0,
        "actual_context_rebuild_ms": 1800.0,
        "wire_ms": 100.0,
        "decode_ms": 800.0,
        "h2d_ms": 50.0,
        "compute_effective_ms": 500.0,
        "physical_dependency_wait_ms": 700.0,
        "support_attention_ms": 25.0,
        "actual_huffman_bitstream": True,
        "runtime_adaptation": True,
        "stream_granularity": "(token,layer,kv-head)",
        "spargeattention": "official Ampere CUDA binding, causal=True",
        "measurement_scope": "preloaded cloud bytes; simulated wireless",
        "stage_records": [
            {
                "stage": 1,
                "predicted_compute_ms": 10.0,
                "actual_compute_worker_ms": 12.0,
                "predicted_stream_ms": 9.0,
                "actual_stream_ms": 11.0,
                "dependency_wait_ms": 3.0,
            }
        ],
    }


def test_loader_ignores_mixed_stdout(tmp_path):
    path = tmp_path / "run.log"
    path.write_text(
        "header\n{\nnot-json\n" + json.dumps(_row(0)) + "\n[SUCCESS]\n",
        encoding="utf-8",
    )
    rows = load_result_records([path])
    assert len(rows) == 1
    assert rows[0]["sample_index"] == 0


def test_generate_plots_and_alignment_audit(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [
        _row(0),
        _row(0, "local_sparse"),
        _row(0, "strong_hybrid"),
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = generate_plots(
        inputs=[path],
        output_dir=tmp_path / "plots",
        formats=["png"],
        max_stage_plots=1,
    )
    assert manifest["result_records"] == 3
    assert (tmp_path / "plots" / "01_ttft.png").is_file()
    assert (tmp_path / "plots" / "03_prediction_gap.png").is_file()
    assert (tmp_path / "plots" / "plot_manifest.json").is_file()

    payload = audit(rows)
    assert payload["records"] == 3
    assert payload["diagnostics"]["median_actual_over_predicted"] == 1.8
    baseline_check = next(
        check for check in payload["checks"]
        if check["area"] == "Paper evaluation baselines"
    )
    assert baseline_check["status"] == "partial"
