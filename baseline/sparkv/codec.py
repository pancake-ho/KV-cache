from __future__ import annotations

import heapq
import io
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

try:
    from bitarray import bitarray
except ImportError as exc:  # pragma: no cover - environment dependency
    raise ImportError(
        "SparKV's actual Huffman bitstream path requires `bitarray`. "
        "Install baseline/sparkv/requirements.txt."
    ) from exc


MAGIC = b"SPKVHUF1"
HEADER_LEN_STRUCT = struct.Struct("<I")


@dataclass(frozen=True)
class QuantizedTensor:
    symbols: np.ndarray
    scale: float
    bits: int
    shape: tuple[int, ...]


def _signed_range(bits: int) -> tuple[int, int, int]:
    if bits < 2 or bits > 8:
        raise ValueError("quantization bits must be in [2, 8]")
    offset = 1 << (bits - 1)
    qmin = -offset
    qmax = offset - 1
    return qmin, qmax, offset


def quantize_symmetric(
    x: torch.Tensor,
    bits: int,
) -> QuantizedTensor:
    """
    Symmetric scalar quantization used for the reproducible SparKV codec.

    The paper's motivation experiment explicitly uses uniform 5-bit
    quantization followed by Huffman coding.  The implementation section later
    states "layer-wise non-uniform quantization" but does not disclose the
    layer-wise bit allocation / codebook.  This codec therefore supports a
    layer-specific bit width, with 5 bits as the paper-grounded default.
    """
    qmin, qmax, offset = _signed_range(bits)

    cpu = (
        x.detach()
        .to(torch.float32)
        .contiguous()
        .cpu()
    )
    max_abs = float(cpu.abs().max().item())

    scale = (
        max_abs / float(qmax)
        if max_abs > 0
        else 1.0
    )

    q = torch.round(cpu / scale).clamp(qmin, qmax)
    symbols = (
        (q.to(torch.int16) + offset)
        .to(torch.uint8)
        .numpy()
        .reshape(-1)
        .copy()
    )

    return QuantizedTensor(
        symbols=symbols,
        scale=scale,
        bits=bits,
        shape=tuple(cpu.shape),
    )


def dequantize_symmetric(
    symbols: np.ndarray,
    *,
    scale: float,
    bits: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    _, _, offset = _signed_range(bits)
    signed = (
        symbols.astype(np.int16)
        - offset
    )
    tensor = torch.from_numpy(
        signed.reshape(shape).copy()
    ).to(torch.float32)
    return (tensor * float(scale)).to(dtype)


def _huffman_code_lengths(
    symbols: np.ndarray,
    alphabet_size: int,
) -> list[int]:
    counts = np.bincount(
        symbols.astype(np.int64),
        minlength=alphabet_size,
    )

    heap: list[
        tuple[int, int, int | tuple]
    ] = []

    serial = 0
    for symbol, count in enumerate(counts):
        if count <= 0:
            continue
        heapq.heappush(
            heap,
            (
                int(count),
                serial,
                int(symbol),
            ),
        )
        serial += 1

    if not heap:
        raise ValueError("cannot Huffman-code an empty tensor")

    lengths = [0] * alphabet_size

    if len(heap) == 1:
        only_symbol = int(heap[0][2])
        lengths[only_symbol] = 1
        return lengths

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        node = (n1, n2)
        heapq.heappush(
            heap,
            (
                f1 + f2,
                serial,
                node,
            ),
        )
        serial += 1

    root = heap[0][2]

    def walk(
        node: int | tuple,
        depth: int,
    ) -> None:
        if isinstance(node, int):
            lengths[node] = max(depth, 1)
            return
        left, right = node
        walk(left, depth + 1)
        walk(right, depth + 1)

    walk(root, 0)
    return lengths


def _canonical_codes(
    lengths: list[int],
) -> dict[int, bitarray]:
    items = sorted(
        (
            (length, symbol)
            for symbol, length in enumerate(lengths)
            if length > 0
        )
    )

    if not items:
        raise ValueError("empty Huffman codebook")

    codes: dict[int, bitarray] = {}
    code = 0
    previous_length = items[0][0]

    for index, (length, symbol) in enumerate(items):
        if index == 0:
            code = 0
        else:
            code += 1
            code <<= (
                length - previous_length
            )

        text = format(
            code,
            f"0{length}b",
        )
        codes[symbol] = bitarray(
            text,
            endian="big",
        )
        previous_length = length

    return codes


def _encode_symbols(
    symbols: np.ndarray,
    bits: int,
) -> tuple[bytes, int, list[int]]:
    alphabet_size = 1 << bits
    lengths = _huffman_code_lengths(
        symbols,
        alphabet_size,
    )
    codes = _canonical_codes(lengths)

    out = bitarray(endian="big")
    out.encode(
        codes,
        (
            int(x)
            for x in symbols
        ),
    )

    bit_length = len(out)
    return (
        out.tobytes(),
        bit_length,
        lengths,
    )


def _decode_symbols(
    payload: bytes,
    *,
    bit_length: int,
    lengths: list[int],
    count: int,
) -> np.ndarray:
    codes = _canonical_codes(lengths)

    bits = bitarray(endian="big")
    bits.frombytes(payload)

    if bit_length < len(bits):
        del bits[bit_length:]

    # bitarray.decode(codebook) is implemented in C and returns an
    # iterator. Consume it directly into the final uint8 array instead of
    # materializing Python int objects first.
    iterator = bits.decode(
        codes
    )
    decoded = np.fromiter(
        iterator,
        dtype=np.uint8,
        count=count,
    )

    if decoded.size != count:
        raise ValueError(
            "Huffman decoded symbol count mismatch: "
            f"expected={count}, got={decoded.size}"
        )

    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError(
            "Huffman decoded more symbols than declared: "
            f"count={count}"
        )

    return decoded


@dataclass(frozen=True)
class EncodedKVUnit:
    header: dict
    k_payload: bytes
    v_payload: bytes

    @property
    def wire_bytes(self) -> int:
        header_bytes = json.dumps(
            self.header,
            separators=(",", ":"),
        ).encode("utf-8")
        return (
            len(MAGIC)
            + HEADER_LEN_STRUCT.size
            + len(header_bytes)
            + len(self.k_payload)
            + len(self.v_payload)
        )


def encode_kv_unit(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    bits: int = 5,
) -> EncodedKVUnit:
    if key.shape != value.shape:
        raise ValueError(
            f"K/V shape mismatch: K={key.shape}, V={value.shape}"
        )

    qk = quantize_symmetric(
        key,
        bits,
    )
    qv = quantize_symmetric(
        value,
        bits,
    )

    k_payload, k_bit_length, k_lengths = (
        _encode_symbols(
            qk.symbols,
            bits,
        )
    )
    v_payload, v_bit_length, v_lengths = (
        _encode_symbols(
            qv.symbols,
            bits,
        )
    )

    header = {
        "version": 1,
        "bits": bits,
        "shape": list(qk.shape),
        "count": int(
            np.prod(qk.shape)
        ),
        "k_scale": float(qk.scale),
        "v_scale": float(qv.scale),
        "k_lengths": k_lengths,
        "v_lengths": v_lengths,
        "k_bit_length": int(
            k_bit_length
        ),
        "v_bit_length": int(
            v_bit_length
        ),
        "k_bytes": len(k_payload),
        "v_bytes": len(v_payload),
    }

    return EncodedKVUnit(
        header=header,
        k_payload=k_payload,
        v_payload=v_payload,
    )


def write_encoded_unit(
    encoded: EncodedKVUnit,
    path: Path,
) -> int:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header_bytes = json.dumps(
        encoded.header,
        separators=(",", ":"),
    ).encode("utf-8")

    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(
            HEADER_LEN_STRUCT.pack(
                len(header_bytes)
            )
        )
        handle.write(header_bytes)
        handle.write(
            encoded.k_payload
        )
        handle.write(
            encoded.v_payload
        )

    return path.stat().st_size


def decode_encoded_bytes(
    blob: bytes,
    *,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict,
]:
    handle = io.BytesIO(
        blob
    )

    magic = handle.read(
        len(MAGIC)
    )
    if magic != MAGIC:
        raise ValueError(
            "invalid SparKV unit magic"
        )

    raw = handle.read(
        HEADER_LEN_STRUCT.size
    )
    if (
        len(raw)
        != HEADER_LEN_STRUCT.size
    ):
        raise ValueError(
            "truncated SparKV unit header"
        )

    (header_len,) = (
        HEADER_LEN_STRUCT.unpack(
            raw
        )
    )

    header_raw = handle.read(
        header_len
    )
    if len(
        header_raw
    ) != header_len:
        raise ValueError(
            "truncated SparKV JSON header"
        )

    header = json.loads(
        header_raw.decode(
            "utf-8"
        )
    )

    k_bytes = int(
        header["k_bytes"]
    )
    v_bytes = int(
        header["v_bytes"]
    )

    k_payload = handle.read(
        k_bytes
    )
    v_payload = handle.read(
        v_bytes
    )

    if (
        len(k_payload)
        != k_bytes
        or len(v_payload)
        != v_bytes
    ):
        raise ValueError(
            "truncated SparKV payload"
        )

    if handle.read(1):
        raise ValueError(
            "unexpected trailing bytes"
        )

    count = int(
        header["count"]
    )
    bits = int(
        header["bits"]
    )
    shape = tuple(
        int(x)
        for x in
        header["shape"]
    )

    k_symbols = _decode_symbols(
        k_payload,
        bit_length=int(
            header[
                "k_bit_length"
            ]
        ),
        lengths=[
            int(x)
            for x in
            header["k_lengths"]
        ],
        count=count,
    )
    v_symbols = _decode_symbols(
        v_payload,
        bit_length=int(
            header[
                "v_bit_length"
            ]
        ),
        lengths=[
            int(x)
            for x in
            header["v_lengths"]
        ],
        count=count,
    )

    key = dequantize_symmetric(
        k_symbols,
        scale=float(
            header["k_scale"]
        ),
        bits=bits,
        shape=shape,
        dtype=dtype,
    )
    value = dequantize_symmetric(
        v_symbols,
        scale=float(
            header["v_scale"]
        ),
        bits=bits,
        shape=shape,
        dtype=dtype,
    )

    return (
        key,
        value,
        header,
    )


def read_encoded_unit(
    path: Path,
    *,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict,
]:
    return decode_encoded_bytes(
        path.read_bytes(),
        dtype=dtype,
    )
