"""5-bit Qwen prefix KV codec adapted from Jungwoo's QKV Bitpack V2.1."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


BITS = 5
CENTER = 15
CODEC_ID = "kv_bitpack_5bit"


@triton.jit
def _encode(K, V, PAYLOAD, SCALES, T, kh, kt, kd, vh, vt, vd,
            HEAD_DIM: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
            BITS: tl.constexpr, CENTER: tl.constexpr):
    projection = tl.program_id(0)
    token = tl.program_id(1)
    channel = tl.arange(0, BLOCK)
    head = channel // HEAD_DIM
    dim = channel % HEAD_DIM
    if projection == 0:
        x = tl.load(K + head * kh + token * kt + dim * kd, channel < WIDTH, other=0)
    else:
        x = tl.load(V + head * vh + token * vt + dim * vd, channel < WIDTH, other=0)
    x = x.to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    safe_maximum = tl.where(maximum > 0, maximum, 1.0)
    # Use the V2.1 scalar factor except where it could overflow in FP32.
    if maximum > 0 and maximum < 1.0e-37:
        scaled = tl.div_rn(x, safe_maximum) * CENTER
    else:
        scaled = x * tl.div_rn(CENTER, safe_maximum)
    q = libdevice.nearbyint(scaled + CENTER).to(tl.uint64)
    q = tl.where(maximum > 0, q, 0)
    q = tl.where(channel < WIDTH, q, 0)

    # Eight 5-bit symbols occupy five little-endian bytes.
    position = tl.arange(0, 8)
    fields = tl.reshape(q, (BLOCK // 8, 8)) << (position[None, :].to(tl.uint64) * BITS)
    word = tl.sum(fields, axis=1)
    group = tl.arange(0, BLOCK // 8)
    byte = group[:, None] * BITS + position[None, :]
    value = (word[:, None] >> (position[None, :].to(tl.uint64) * 8)) & 255
    row = projection * T + token
    valid = (group[:, None] < WIDTH // 8) & (position[None, :] < BITS)
    tl.store(PAYLOAD + row * (WIDTH * BITS // 8) + byte, value.to(tl.uint8), valid)
    tl.store(SCALES + row, maximum)


@triton.jit
def _decode(PAYLOAD, SCALES, OUT, T, HEADS: tl.constexpr,
            HEAD_DIM: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
            BITS: tl.constexpr, CENTER: tl.constexpr):
    projection = tl.program_id(1)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = T * WIDTH
    nbytes = n * BITS // 8
    byte = i * BITS // 8
    shift = i * BITS % 8
    first = tl.load(PAYLOAD + projection * nbytes + byte, i < n, other=0).to(tl.uint32)
    second = tl.load(PAYLOAD + projection * nbytes + byte + 1,
                     (i < n) & (byte + 1 < nbytes), other=0).to(tl.uint32)
    q = ((first | (second << 8)) >> shift) & ((1 << BITS) - 1)
    token = i // WIDTH
    scale = tl.load(SCALES + projection * T + token, i < n, other=0)
    value = tl.div_rn(q.to(tl.float32) - CENTER, CENTER) * scale
    channel = i % WIDTH
    head = channel // HEAD_DIM
    dim = channel % HEAD_DIM
    offset = ((projection * HEADS + head) * T + token) * HEAD_DIM + dim
    tl.store(OUT + offset, value, i < n)


@dataclass(frozen=True)
class PackedLayer:
    payload: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, int, int, int]
    dtype: torch.dtype


class KVBitpackCodec:
    """Pack finite CUDA K/V layers [1, heads, tokens, head_dim] independently."""

    @staticmethod
    def encoded_size(layers):
        return sum(2 * k.shape[2] * (k.shape[1] * k.shape[3] * BITS // 8 + 4)
                   for k, _ in layers)

    @staticmethod
    @torch.inference_mode()
    def encode(layers):
        encoded = []
        for k, v in layers:
            if (k.ndim != 4 or k.shape[0] != 1 or min(k.shape) < 1 or k.shape != v.shape
                    or not k.is_cuda or k.device != v.device or k.dtype != v.dtype
                    or k.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
                raise ValueError("K/V must have the same CUDA shape [1, H, T, D] and floating dtype")
            _, heads, tokens, head_dim = k.shape
            width = heads * head_dim
            if width % 8:
                raise ValueError("5-bit packing requires H * D to be divisible by 8")
            payload = torch.empty((2, tokens, width * BITS // 8), dtype=torch.uint8, device=k.device)
            scales = torch.empty((2, tokens), dtype=torch.float32, device=k.device)
            with torch.cuda.device(k.device):
                _encode[(2, tokens)](
                    k, v, payload, scales, tokens, *k.stride()[1:], *v.stride()[1:],
                    HEAD_DIM=head_dim, WIDTH=width, BLOCK=triton.next_power_of_2(width),
                    BITS=BITS, CENTER=CENTER, num_warps=4, enable_fp_fusion=False,
                )
            encoded.append(PackedLayer(payload, scales, tuple(k.shape), k.dtype))
        return tuple(encoded)

    @staticmethod
    @torch.inference_mode()
    def decode(layers):
        decoded = []
        for layer in layers:
            _, heads, tokens, head_dim = layer.shape
            width = heads * head_dim
            device = layer.payload.device
            out = torch.empty((2, *layer.shape), dtype=layer.dtype, device=device)
            with torch.cuda.device(device):
                _decode[(triton.cdiv(tokens * width, 256), 2)](
                    layer.payload, layer.scales, out, tokens, HEADS=heads,
                    HEAD_DIM=head_dim, WIDTH=width, BLOCK=256,
                    BITS=BITS, CENTER=CENTER, num_warps=4, enable_fp_fusion=False,
                )
            decoded.append(out.unbind(0))
        return tuple(decoded)
