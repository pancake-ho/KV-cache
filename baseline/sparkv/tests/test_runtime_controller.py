from baseline.sparkv.runtime_controller import (
    RuntimeController,
    RuntimeControllerConfig,
)
from baseline.sparkv.scheduler import Chunk


def _stage():
    return {
        "stage": 1,
        "compute": [
            {"t": 0, "layer": 0, "head": 0},
            {"t": 0, "layer": 0, "head": 1},
        ],
        "stream": [
            {"t": 1, "layer": 0, "head": 0},
        ],
    }


def test_compute_contention_moves_tail_compute_to_stream():
    controller = RuntimeController(
        layers=2,
        config=RuntimeControllerConfig(
            window=2,
            imbalance_margin=0.0,
            max_migrations_per_stage=1,
        ),
    )

    controller.observe_stage(
        predicted_compute_ms=1.0,
        actual_compute_ms=2.0,
        predicted_stream_ms=1.0,
        actual_stream_ms=1.0,
    )

    current, _, log = controller.adapt(
        current_stage=_stage(),
        next_stage=None,
        done={},
    )

    assert log["migrations"] == 1
    assert len(current["compute"]) == 1
    assert len(current["stream"]) == 2


def test_bandwidth_bottleneck_moves_ready_stream_to_compute():
    controller = RuntimeController(
        layers=2,
        config=RuntimeControllerConfig(
            window=2,
            imbalance_margin=0.0,
            max_migrations_per_stage=1,
        ),
    )

    controller.observe_stage(
        predicted_compute_ms=1.0,
        actual_compute_ms=1.0,
        predicted_stream_ms=1.0,
        actual_stream_ms=2.0,
    )

    current, _, log = controller.adapt(
        current_stage=_stage(),
        next_stage=None,
        done={
            Chunk(0, 0, 0): "stream",
        },
    )

    assert log["migrations"] == 1
    assert any(
        item["t"] == 1
        for item in current["compute"]
    )


def test_layer_dependency_requires_previous_local_compute():
    controller = RuntimeController(
        layers=3,
        config=RuntimeControllerConfig(),
    )
    c = Chunk(0, 1, 0)
    previous = Chunk(0, 0, 0)

    assert not controller.compute_ready(
        c,
        {previous: "stream"},
    )
    assert controller.compute_ready(
        c,
        {previous: "compute"},
    )
