import copy

import pytest
import torch

from baseline.sparkv.executor import (
    _append_token,
    _schedule_assignments,
    _validate_against_meta,
)

from baseline.sparkv.scheduler import (
    Chunk,
    SparKVScheduler,
)


def make_scheduler(
    *,
    token_chunks: int = 2,
    layers: int = 3,
    heads: int = 2,
    delta_ms: float = 2.0,
):
    chunks = [
        Chunk(
            t,
            layer,
            head,
        )
        for t in range(
            token_chunks
        )
        for layer in range(
            layers
        )
        for head in range(
            heads
        )
    ]

    comp = {
        chunk: 1.0
        for chunk in chunks
    }

    stream = {
        chunk: 1.2
        for chunk in chunks
    }

    return SparKVScheduler(
        token_chunks=token_chunks,
        layers=layers,
        heads=heads,
        comp_ms=comp,
        stream_ms=stream,
        delta_ms=delta_ms,
    )


def make_meta(
    *,
    token_chunks: int,
    layers: int,
    heads: int,
    chunk_size: int = 3,
):
    seq_len = (
        token_chunks
        * chunk_size
    )

    wire_bytes_per_unit = 16

    chunks = []

    unit_files = {}

    for t in range(
        token_chunks
    ):
        lh_wire_bytes = {
            f"{layer}:{head}":
                wire_bytes_per_unit
            for layer in range(
                layers
            )
            for head in range(
                heads
            )
        }

        chunks.append(
            {
                "index": t,

                "wire_bytes": (
                    layers
                    * heads
                    * wire_bytes_per_unit
                ),

                "lh_wire_bytes":
                    lh_wire_bytes,
            }
        )

        for layer in range(
            layers
        ):
            for head in range(
                heads
            ):
                key = (
                    f"{t}:"
                    f"{layer}:"
                    f"{head}"
                )

                unit_files[key] = {
                    "path": (
                        f"units/"
                        f"t{t:03d}/"
                        f"l{layer:02d}_"
                        f"h{head:02d}"
                        f".safetensors"
                    ),

                    "wire_bytes":
                        wire_bytes_per_unit,

                    # Synthetic metadata only.
                    # Actual value is produced by
                    # unit_cache.materialize_sample().
                    "storage_bytes":
                        wire_bytes_per_unit,
                }

    return {
        "seq_len": seq_len,

        "chunk_size":
            chunk_size,

        "num_chunks":
            token_chunks,

        "layers":
            layers,

        "kv_heads":
            heads,

        "chunks":
            chunks,

        # P1 fine-grained streaming
        # metadata contract.
        "unit_layout_version": 1,

        "unit_files":
            unit_files,

        "unit_file_count":
            len(unit_files),

        "unit_wire_bytes": (
            token_chunks
            * layers
            * heads
            * wire_bytes_per_unit
        ),
    }


def test_schedule_contract_matches_cache_geometry():
    scheduler = make_scheduler(
        token_chunks=2,
        layers=2,
        heads=2,
    )

    schedule = scheduler.run()

    meta = make_meta(
        token_chunks=2,
        layers=2,
        heads=2,
    )

    assignments, stream_order = (
        _validate_against_meta(
            schedule,
            meta,
        )
    )

    assert len(assignments) == 8

    assert set(
        assignments.values()
    ) <= {
        "compute",
        "stream",
    }

    assert len(stream_order) == sum(
        route == "stream"
        for route in assignments.values()
    )

    assert all(
        assignments[chunk]
        == "stream"
        for chunk in stream_order
    )


def test_schedule_contains_every_unit_exactly_once():
    scheduler = make_scheduler(
        token_chunks=3,
        layers=2,
        heads=2,
    )

    schedule = scheduler.run()

    assignments, _ = (
        _schedule_assignments(
            schedule
        )
    )

    expected = {
        Chunk(
            t,
            layer,
            head,
        )
        for t in range(3)
        for layer in range(2)
        for head in range(2)
    }

    assert set(assignments) == expected

    assert (
        len(assignments)
        == 3 * 2 * 2
    )


def test_duplicate_scheduled_unit_is_rejected():
    scheduler = make_scheduler(
        token_chunks=2,
        layers=2,
        heads=1,
    )

    schedule = scheduler.run()

    duplicate = copy.deepcopy(
        schedule
    )

    source = None

    for stage in duplicate[
        "stages"
    ]:
        for operation in (
            "compute",
            "stream",
        ):
            if stage[operation]:
                source = copy.deepcopy(
                    stage[operation][0]
                )

                stage[
                    operation
                ].append(source)

                duplicate["chunks"] += 1

                if (
                    operation
                    == "compute"
                ):
                    duplicate[
                        "compute_chunks"
                    ] += 1
                else:
                    duplicate[
                        "stream_chunks"
                    ] += 1

                break

        if source is not None:
            break

    assert source is not None

    with pytest.raises(
        ValueError,
        match="duplicate scheduled unit",
    ):
        _schedule_assignments(
            duplicate
        )


def test_geometry_mismatch_is_rejected():
    scheduler = make_scheduler(
        token_chunks=2,
        layers=2,
        heads=2,
    )

    schedule = scheduler.run()

    wrong_meta = make_meta(
        token_chunks=2,
        layers=2,
        heads=3,
    )

    with pytest.raises(
        ValueError,
        match=(
            "schedule/cache geometry "
            "mismatch"
        ),
    ):
        _validate_against_meta(
            schedule,
            wrong_meta,
        )


def test_missing_wire_metadata_is_rejected():
    scheduler = make_scheduler(
        token_chunks=2,
        layers=2,
        heads=2,
    )

    schedule = scheduler.run()

    meta = make_meta(
        token_chunks=2,
        layers=2,
        heads=2,
    )

    del meta[
        "chunks"
    ][0][
        "lh_wire_bytes"
    ][
        "1:1"
    ]

    with pytest.raises(
        ValueError,
        match="missing wire size",
    ):
        _validate_against_meta(
            schedule,
            meta,
        )


def test_append_token_preserves_token_and_head_order():
    layers = 2
    heads = 2
    chunk_size = 3

    first = {}
    second = {}

    for layer in range(layers):
        for head in range(heads):
            a = float(
                10 * layer
                + head
            )

            b = float(
                100
                + 10 * layer
                + head
            )

            first[
                Chunk(
                    0,
                    layer,
                    head,
                )
            ] = (
                torch.full(
                    (
                        1,
                        1,
                        chunk_size,
                        4,
                    ),
                    a,
                ),

                torch.full(
                    (
                        1,
                        1,
                        chunk_size,
                        4,
                    ),
                    a + 0.5,
                ),
            )

            second[
                Chunk(
                    1,
                    layer,
                    head,
                )
            ] = (
                torch.full(
                    (
                        1,
                        1,
                        chunk_size,
                        4,
                    ),
                    b,
                ),

                torch.full(
                    (
                        1,
                        1,
                        chunk_size,
                        4,
                    ),
                    b + 0.5,
                ),
            )

    cache = _append_token(
        None,
        first,
        t=0,
        layers=layers,
        heads=heads,
        chunk_size=chunk_size,
        to_legacy=lambda x:
            x.to_legacy_cache(),
    )

    cache = _append_token(
        cache,
        second,
        t=1,
        layers=layers,
        heads=heads,
        chunk_size=chunk_size,
        to_legacy=lambda x:
            x.to_legacy_cache(),
    )

    assert (
        cache.get_seq_length()
        == 2 * chunk_size
    )

    legacy = (
        cache.to_legacy_cache()
    )

    key = legacy[0][0]

    assert key.shape == (
        1,
        heads,
        2 * chunk_size,
        4,
    )

    assert torch.all(
        key[
            :,
            0,
            :chunk_size,
        ]
        == 0.0
    )

    assert torch.all(
        key[
            :,
            1,
            :chunk_size,
        ]
        == 1.0
    )

    assert torch.all(
        key[
            :,
            0,
            chunk_size:,
        ]
        == 100.0
    )

    assert torch.all(
        key[
            :,
            1,
            chunk_size:,
        ]
        == 101.0
    )


def test_append_token_rejects_missing_head():
    chunk_size = 3

    units = {
        Chunk(
            0,
            0,
            0,
        ): (
            torch.zeros(
                1,
                1,
                chunk_size,
                4,
            ),

            torch.zeros(
                1,
                1,
                chunk_size,
                4,
            ),
        )
    }

    with pytest.raises(
        RuntimeError,
        match="missing KV unit",
    ):
        _append_token(
            None,
            units,
            t=0,
            layers=1,
            heads=2,
            chunk_size=chunk_size,
            to_legacy=lambda x:
                x.to_legacy_cache(),
        )