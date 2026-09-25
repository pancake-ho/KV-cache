import numpy as np
import torch

from baseline.sparkv.codec import (
    dequantize_symmetric,
    quantize_symmetric,
)


def test_symmetric_5bit_range_shape_and_round_trip():
    torch.manual_seed(11)

    x = torch.randn(
        1,
        1,
        64,
        128,
        dtype=torch.float32,
    )

    quantized = quantize_symmetric(
        x,
        bits=5,
    )

    assert quantized.bits == 5
    assert quantized.shape == tuple(
        x.shape
    )
    assert quantized.symbols.dtype == (
        np.uint8
    )
    assert int(
        quantized.symbols.min()
    ) >= 0
    assert int(
        quantized.symbols.max()
    ) <= 31

    recovered = dequantize_symmetric(
        quantized.symbols,
        scale=quantized.scale,
        bits=quantized.bits,
        shape=quantized.shape,
        dtype=torch.float32,
    )

    assert recovered.shape == x.shape
    assert torch.isfinite(
        recovered
    ).all()

    # Nearest rounding under the symmetric scalar quantizer is bounded by
    # approximately half a quantization step, aside from endpoint clipping.
    max_error = (
        x - recovered
    ).abs().max().item()

    assert max_error <= (
        quantized.scale
        * 0.55
        + 1e-6
    )
