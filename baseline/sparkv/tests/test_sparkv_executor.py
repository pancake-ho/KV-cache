import torch

from baseline.sparkv.scheduler import (
    Chunk,
    SparKVScheduler,
)

from baseline.sparkv.sparkv_executor import (
    _append_token,
    _validate_against_meta,
)


def make_schedule():
    chunks = [
        Chunk(t, layer, head)
        for t in range(2)
        for layer in range(2)
        for head in range(2)
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
        token_chunks=2,
        layers=2,
        heads=2,
        comp_ms=comp,
        stream_ms=stream,
        delta_ms=2.0,
    ).run()


def test_schedule_contract_matches_cache_geometry():
    schedule = make_schedule()

    meta = {
        "num_chunks": 2,
        "layers": 2,
        "kv_heads": 2,
    }

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

    assert all(
        assignments[c] == "stream"
        for c in stream_order
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
                10 * layer + head
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
        to_legacy=lambda x: (
            x.to_legacy_cache()
        ),
    )

    cache = _append_token(
        cache,
        second,
        t=1,
        layers=layers,
        heads=heads,
        chunk_size=chunk_size,
        to_legacy=lambda x: (
            x.to_legacy_cache()
        ),
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