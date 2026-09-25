import json

from baseline.sparkv.evaluation import (
    build_all_stream_schedule,
    build_local_sparse_schedule,
    build_strong_hybrid_schedule,
)


def test_all_stream_control_preserves_costs(tmp_path):
    source = {
        "makespan_ms": 7.0,
        "chunks": 2,
        "compute_chunks": 1,
        "stream_chunks": 1,
        "stages": [],
        "unit_costs": {
            "0:0:0": {
                "comp_ms": 1.0,
                "stream_ms": 2.5,
            },
            "0:0:1": {
                "comp_ms": 4.0,
                "stream_ms": 3.5,
            },
        },
    }
    src = (
        tmp_path
        / "sparkv.json"
    )
    dst = (
        tmp_path
        / "all_stream.json"
    )
    src.write_text(
        json.dumps(source),
        encoding="utf-8",
    )

    result = (
        build_all_stream_schedule(
            sparkv_schedule_path=src,
            output=dst,
        )
    )

    assert result[
        "compute_chunks"
    ] == 0
    assert result[
        "stream_chunks"
    ] == 2
    assert result[
        "makespan_ms"
    ] == 6.0
    assert result[
        "assignments"
    ] == {
        "0:0:0": "stream",
        "0:0:1": "stream",
    }
    assert len(
        result[
            "stages"
        ]
    ) == 1
    assert len(
        result[
            "stages"
        ][0][
            "stream"
        ]
    ) == 2


def _fixed_source():
    return {
        "delta_ms": 2.0,
        "unit_costs": {
            f"{token}:{layer}:0": {
                "comp_ms": 1.0,
                "stream_ms": 2.0,
            }
            for token in range(2)
            for layer in range(2)
        },
    }


def test_local_sparse_uses_same_compute_path_for_every_unit(tmp_path):
    src = tmp_path / "sparkv.json"
    src.write_text(json.dumps(_fixed_source()), encoding="utf-8")

    result = build_local_sparse_schedule(
        sparkv_schedule_path=src,
        output=tmp_path / "local.json",
    )

    assert result["compute_chunks"] == 4
    assert result["stream_chunks"] == 0
    assert set(result["assignments"].values()) == {"compute"}
    assert result["same_spargeattention_as_sparkv"] is True


def test_strong_hybrid_is_early_compute_later_stream(tmp_path):
    src = tmp_path / "sparkv.json"
    src.write_text(json.dumps(_fixed_source()), encoding="utf-8")

    result = build_strong_hybrid_schedule(
        sparkv_schedule_path=src,
        output=tmp_path / "hybrid.json",
        compute_fraction=0.5,
    )

    assert result["compute_chunks"] == 2
    assert result["stream_chunks"] == 2
    assert result["assignments"]["0:0:0"] == "compute"
    assert result["assignments"]["0:1:0"] == "compute"
    assert result["assignments"]["1:0:0"] == "stream"
    assert result["assignments"]["1:1:0"] == "stream"
    assert result["partition_parameter_paper_disclosed"] is False
