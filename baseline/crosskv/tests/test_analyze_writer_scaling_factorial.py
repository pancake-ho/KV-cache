from xmodel_kv.cli.analyze_writer_scaling_factorial import analyze_factorial


def _row(case_id, *, exact, f1):
    return {
        "id": case_id,
        "student_exact": exact,
        "student_f1": f1,
        "bridge_exact": exact,
        "bridge_f1": f1,
        "no_summary_exact": 0,
        "no_summary_f1": 0,
        "active_layers": 16,
        "wire_payload_bytes": 10,
    }


def _table(high):
    table = {}
    for layer in ("all", "last_16"):
        for quant in (16, 4):
            score = high if layer == "all" and quant == 16 else 0
            table[(layer, quant)] = {
                "a": _row("a", exact=score, f1=score),
                "b": _row("b", exact=score, f1=score),
            }
    return table


def test_factorial_reports_writer_and_within_writer_contrasts():
    result = analyze_factorial(
        _table(0),
        _table(1),
        bootstrap_replicates=100,
        seed=1,
    )
    assert result["cases"] == 2
    assert result["cells"]["n512"]["all_bf16"]["mean_f1"] == 1
    assert result["n512_vs_n128"]["all_bf16"]["mean_f1_delta"] == 1
    assert (
        result["within_writer"]["n512"]["last_16_vs_all_bf16"][
            "mean_f1_delta"
        ]
        == -1
    )


def test_factorial_rejects_writer_id_mismatch():
    n128 = _table(0)
    n512 = _table(1)
    del n512[("all", 16)]["a"]
    try:
        analyze_factorial(n128, n512, bootstrap_replicates=10, seed=1)
    except ValueError as error:
        assert "IDs differ" in str(error)
    else:
        raise AssertionError("expected ID mismatch")


def test_factorial_supports_semantic_writer_names():
    result = analyze_factorial(
        _table(0),
        _table(1),
        bootstrap_replicates=10,
        seed=1,
        reference_name="unconstrained",
        candidate_name="packet_aware",
    )
    assert "packet_aware_vs_unconstrained" in result
    assert "packet_aware" in result["cells"]
