import pytest

from baseline.sparkv.scheduler import (
    Chunk,
    SparKVScheduler,
    resolve_delta_ms,
)


def make_scheduler(
    t=2,
    layers=3,
    heads=2,
    delta=2.0,
):
    chunks = [
        Chunk(i, layer, head)
        for i in range(t)
        for layer in range(layers)
        for head in range(heads)
    ]
    comp = {c: 1.0 for c in chunks}
    stream = {c: 1.2 for c in chunks}
    return SparKVScheduler(
        t,
        layers,
        heads,
        comp,
        stream,
        delta,
    )


def test_each_chunk_is_processed_exactly_once():
    scheduler = make_scheduler()
    result = scheduler.run()
    scheduled = []

    for stage in result["stages"]:
        scheduled.extend(
            ("compute", tuple(x.values()))
            for x in stage["compute"]
        )
        scheduled.extend(
            ("stream", tuple(x.values()))
            for x in stage["stream"]
        )

    assert len(scheduled) == result["chunks"]
    assert len({chunk for _, chunk in scheduled}) == result["chunks"]


def test_vertical_dependency_requires_local_compute():
    scheduler = make_scheduler(
        t=1,
        layers=2,
        heads=1,
    )
    parent = Chunk(0, 0, 0)
    child = Chunk(0, 1, 0)

    scheduler.done[parent] = "stream"
    assert not scheduler.compute_ready(child)

    scheduler.done[parent] = "compute"
    assert scheduler.compute_ready(child)


def test_token_dependency_accepts_either_path():
    scheduler = make_scheduler(
        t=2,
        layers=2,
        heads=1,
    )
    current = Chunk(1, 0, 0)
    previous = Chunk(0, 0, 0)

    scheduler.done[previous] = "stream"
    assert scheduler.compute_ready(current)


def test_too_small_explicit_delta_fails_without_oversized_exception():
    chunks = [Chunk(0, 0, 0)]
    comp = {chunks[0]: 13.0}
    stream = {chunks[0]: 31.0}

    scheduler = SparKVScheduler(
        1,
        1,
        1,
        comp,
        stream,
        5.0,
    )

    with pytest.raises(
        RuntimeError,
        match="made no progress",
    ):
        scheduler.run()


def test_auto_delta_uses_safe_stream_bound_and_completes():
    chunks = [
        Chunk(0, 0, 0),
        Chunk(0, 0, 1),
    ]
    comp = {
        chunks[0]: 13.0,
        chunks[1]: 14.0,
    }
    stream = {
        chunks[0]: 31.0,
        chunks[1]: 37.0,
    }

    delta_ms, meta = resolve_delta_ms(
        "auto",
        comp_ms=comp,
        stream_ms=stream,
    )

    assert delta_ms == 37.0
    assert meta["mode"] == "auto-safe-stream-bound"
    assert meta["paper_disclosed"] is False

    scheduler = SparKVScheduler(
        1,
        1,
        2,
        comp,
        stream,
        delta_ms,
    )
    result = scheduler.run()

    assert result["chunks"] == 2
    assert len(result["assignments"]) == 2


def test_explicit_delta_is_preserved():
    c = Chunk(0, 0, 0)
    delta_ms, meta = resolve_delta_ms(
        "42.5",
        comp_ms={c: 10.0},
        stream_ms={c: 20.0},
    )

    assert delta_ms == 42.5
    assert meta["mode"] == "explicit"
    assert meta["paper_disclosed"] is False
