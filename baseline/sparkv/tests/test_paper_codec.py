import torch

from baseline.sparkv.codec import (
    decode_encoded_bytes,
    encode_kv_unit,
    HEADER_LEN_STRUCT,
    MAGIC,
)


def _blob(encoded):
    import json

    header = json.dumps(
        encoded.header,
        separators=(",", ":"),
    ).encode("utf-8")

    return (
        MAGIC
        + HEADER_LEN_STRUCT.pack(
            len(header)
        )
        + header
        + encoded.k_payload
        + encoded.v_payload
    )


def test_actual_huffman_round_trip_shape_and_finite():
    torch.manual_seed(7)
    key = torch.randn(
        1,
        1,
        32,
        16,
        dtype=torch.float32,
    )
    value = torch.randn_like(
        key
    )

    encoded = encode_kv_unit(
        key,
        value,
        bits=5,
    )

    key2, value2, header = (
        decode_encoded_bytes(
            _blob(encoded),
            dtype=torch.float32,
        )
    )

    assert key2.shape == key.shape
    assert value2.shape == value.shape
    assert torch.isfinite(key2).all()
    assert torch.isfinite(value2).all()
    assert header["bits"] == 5

    # Symmetric rounding error should be bounded by roughly half a scale.
    assert (
        (key - key2)
        .abs()
        .max()
        .item()
        <= float(
            header["k_scale"]
        )
        * 0.55
        + 1e-6
    )


def test_constant_tensor_is_huffman_compressible():
    key = torch.zeros(
        1,
        1,
        64,
        32,
    )
    value = torch.zeros_like(
        key
    )

    encoded = encode_kv_unit(
        key,
        value,
        bits=5,
    )

    raw_bytes = (
        key.numel()
        + value.numel()
    ) * key.element_size()

    # Include header overhead and still expect a constant symbol stream to
    # compress below float32 storage.
    assert (
        encoded.wire_bytes
        < raw_bytes
    )
