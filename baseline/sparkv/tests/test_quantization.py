import torch

from baseline.sparkv.experiment import huffman_payload_bytes, quantize_uniform_5bit


def test_uniform_5bit_range_and_shape():
    x = torch.randn(1, 8, 64, 128, dtype=torch.bfloat16)
    q, scale = quantize_uniform_5bit(x)
    assert q.dtype == torch.uint8
    assert q.shape == x.shape
    assert scale.shape == (1, 8, 1, 1)
    assert int(q.min()) >= 0
    assert int(q.max()) <= 31


def test_huffman_payload_is_positive_and_smaller_for_constant_symbols():
    constant = torch.zeros(1024, dtype=torch.uint8)
    varied = torch.arange(32, dtype=torch.uint8).repeat(32)
    assert huffman_payload_bytes(constant) > 0
    assert huffman_payload_bytes(constant) < huffman_payload_bytes(varied)

