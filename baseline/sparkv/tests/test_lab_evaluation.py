import json

from baseline.sparkv.evaluation import (
    build_all_stream_schedule,
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
