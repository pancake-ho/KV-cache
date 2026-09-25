from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

try:
    import spas_sage_attn._qattn as _qattn
    from spas_sage_attn.utils import (
        block_map_lut_triton,
        get_block_map_meansim_fuse_quant,
        get_vanilla_qk_quant,
        hyperparameter_check,
    )
except ImportError as exc:  # pragma: no cover - environment dependency
    raise ImportError(
        "SparKV requires the official thu-ml/SpargeAttn package. "
        "Install it before running the paper path."
    ) from exc


QUERY_BLOCK = 128
KEY_BLOCK = 64
ATTENTION_MASS_CDF = 0.98


def repeat_kv(
    hidden_states: torch.Tensor,
    n_rep: int,
) -> torch.Tensor:
    batch, kv_heads, length, head_dim = (
        hidden_states.shape
    )
    if n_rep == 1:
        return hidden_states

    expanded = hidden_states[
        :, :, None, :, :
    ].expand(
        batch,
        kv_heads,
        n_rep,
        length,
        head_dim,
    )
    return expanded.reshape(
        batch,
        kv_heads * n_rep,
        length,
        head_dim,
    )


def _require_ampere_kernel(
    device: torch.device,
) -> None:
    if device.type != "cuda":
        raise RuntimeError(
            "SpargeAttention paper path requires CUDA."
        )

    major, minor = torch.cuda.get_device_capability(
        device
    )
    arch = f"sm{major}{minor}"

    # The exact wrapper below uses the official FP16-value Ampere
    # binding.  It covers the current RTX 30 / A100-class path.
    if arch not in {
        "sm80",
        "sm86",
        "sm87",
    }:
        raise RuntimeError(
            "This direct SparKV reproduction currently targets "
            "the official SpargeAttention Ampere kernel "
            f"(sm80/sm86/sm87), but detected {arch}. "
            "Do not silently fall back to SDPA."
        )


def gpu_utilization_percent(
    device_index: int = 0,
) -> float:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(
            int(device_index)
        )
        util = pynvml.nvmlDeviceGetUtilizationRates(
            handle
        )
        return float(util.gpu)
    except Exception:
        # The predictor input is explicitly GPU utilization in the paper.
        # Failing loudly is preferable for profiling; runtime may use 0 only
        # if NVML is unavailable and the caller explicitly accepts it.
        return 0.0


@dataclass(frozen=True)
class SparseAttentionResult:
    output: torch.Tensor
    active_blocks: int
    elapsed_ms: float
    sparsity: float


def _build_mass_mask(
    q_full: torch.Tensor,
    k_full: torch.Tensor,
) -> torch.Tensor:
    """
    Reuse SpargeAttention's own block-map construction.

    q_full and k_full have identical sequence length so that the official
    causal mask is aligned with global token positions.  Only the current
    token chunk of q_full contains real queries; earlier query rows are zero
    padding and their outputs are discarded.  This preserves the suffix
    position offset while using the official causal CUDA kernel.
    """
    if q_full.shape[-2] != k_full.shape[-2]:
        raise ValueError(
            "q_full and k_full must have equal global sequence length"
        )

    km = k_full.mean(
        dim=-2,
        keepdim=True,
    )

    final_map, *_ = (
        get_block_map_meansim_fuse_quant(
            q_full,
            k_full,
            km,
            is_causal=True,
            BLKQ=QUERY_BLOCK,
            BLKK=KEY_BLOCK,
            simthreshd1=0.1,
            cdfthreshd=ATTENTION_MASS_CDF,
            topk=None,
            is_sparse=True,
            return_lut=False,
            attention_sink=False,
        )
    )

    return final_map.contiguous()


def _ampere_block_sparse_causal(
    q_full: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    block_map: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Directly invoke the official SpargeAttention Ampere binding while setting
    is_causal=1.  The public block_sparse_sage2_attn_cuda wrapper hard-codes
    is_causal=False; SparKV is causal prefill, so we expose the underlying
    binding's causal flag from the official attn_cuda.h signature.
    """
    _require_ampere_kernel(
        q_full.device
    )

    if q_full.shape != k_full.shape:
        raise ValueError(
            "Q/K shapes must match after GQA KV repeat"
        )
    if k_full.shape != v_full.shape:
        raise ValueError(
            "K/V shapes must match after GQA KV repeat"
        )

    dtype = q_full.dtype
    if dtype in {
        torch.float32,
        torch.float16,
    }:
        q_kernel = (
            q_full.contiguous()
            .to(torch.float16)
        )
        k_kernel = (
            k_full.contiguous()
            .to(torch.float16)
        )
        v_kernel = (
            v_full.contiguous()
            .to(torch.float16)
        )
    else:
        q_kernel = (
            q_full.contiguous()
            .to(torch.bfloat16)
        )
        k_kernel = (
            k_full.contiguous()
            .to(torch.bfloat16)
        )
        v_kernel = (
            v_full.contiguous()
            .to(torch.float16)
        )

    km = k_kernel.mean(
        dim=-2,
        keepdim=True,
    )

    q_int8, q_scale, k_int8, k_scale = (
        get_vanilla_qk_quant(
            q_kernel,
            k_kernel,
            km,
            QUERY_BLOCK,
            KEY_BLOCK,
        )
    )

    lut, valid_block_num = (
        block_map_lut_triton(
            block_map
        )
    )

    pv_threshold = hyperparameter_check(
        50,
        q_kernel.size(-3),
        q_kernel.device,
    )

    out = torch.empty_like(
        q_kernel
    )

    _qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
        q_int8,
        k_int8,
        v_kernel,
        out,
        lut,
        valid_block_num,
        pv_threshold,
        q_scale,
        k_scale,
        1,      # HND
        True,   # causal prefill
        1,      # qk quant granularity
        float(scale),
        0,      # no pv-count return
    )

    return out



def attention_mask_active_blocks(
    *,
    query_current: torch.Tensor,
    key_history_and_current: torch.Tensor,
    current_chunk_tokens: int,
    num_key_value_groups: int,
) -> tuple[int, float]:
    """
    Compute SparKV predictor feature s without running the attention kernel.
    """
    q_len = int(
        query_current.shape[-2]
    )
    if q_len != current_chunk_tokens:
        raise ValueError(
            "query length does not match current chunk"
        )

    key_full = repeat_kv(
        key_history_and_current,
        num_key_value_groups,
    )
    kv_len = int(
        key_full.shape[-2]
    )
    prefix = kv_len - q_len
    if prefix < 0:
        raise ValueError(
            "KV shorter than current query"
        )

    if prefix:
        prefix_fill = (
            query_current[
                ...,
                :1,
                :,
            ]
            .expand(
                query_current.shape[0],
                query_current.shape[1],
                prefix,
                query_current.shape[-1],
            )
            .contiguous()
        )
        q_full = torch.cat(
            [
                prefix_fill,
                query_current,
            ],
            dim=-2,
        )
    else:
        q_full = query_current

    block_map = _build_mass_mask(
        q_full,
        key_full,
    )
    current_q_blocks = math.ceil(
        q_len / QUERY_BLOCK
    )
    current_map = block_map[
        ...,
        -current_q_blocks:,
        :,
    ]
    active = int(
        current_map.sum().item()
    )
    total = max(
        int(current_map.numel()),
        1,
    )
    return (
        active,
        1.0
        - active / total,
    )


def sparse_attention_current_chunk(
    *,
    query_current: torch.Tensor,
    key_history_and_current: torch.Tensor,
    value_history_and_current: torch.Tensor,
    current_chunk_tokens: int,
    num_key_value_groups: int,
    scale: float,
) -> SparseAttentionResult:
    """
    Execute SparKV's block-sparse attention for one token chunk.

    Shapes before GQA repeat:
      query_current: [B, num_attention_heads, 1024, D]
      key/value:     [B, num_kv_heads, (t+1)*1024, D]

    The paper's feature t corresponds to global query length 1024*t.  To keep
    the official causal kernel globally aligned for a suffix chunk, we prepend
    zero query rows up to the current position, run the causal block-sparse
    kernel, and retain only the current 1024-token output.
    """
    if current_chunk_tokens <= 0:
        raise ValueError(
            "current_chunk_tokens must be positive"
        )

    kv_len = int(
        key_history_and_current.shape[-2]
    )
    q_len = int(
        query_current.shape[-2]
    )

    if q_len != current_chunk_tokens:
        raise ValueError(
            "query_current length must equal current chunk size"
        )

    if kv_len < q_len:
        raise ValueError(
            "KV length cannot be shorter than current Q length"
        )

    key_full = repeat_kv(
        key_history_and_current,
        num_key_value_groups,
    )
    value_full = repeat_kv(
        value_history_and_current,
        num_key_value_groups,
    )

    attention_heads = int(
        query_current.shape[1]
    )
    if key_full.shape[1] != attention_heads:
        raise ValueError(
            "GQA-expanded KV head count does not match Q heads"
        )

    prefix = kv_len - q_len
    if prefix:
        # SpargeAttention's mask builder normalizes query rows and does not
        # add an epsilon to zero-norm blocks.  Fill the discarded historical
        # query prefix with a real query vector rather than zeros so mask
        # construction remains numerically finite.  These prefix outputs are
        # discarded; the suffix positions remain globally aligned for the
        # causal kernel.
        prefix_fill = (
            query_current[
                ...,
                :1,
                :,
            ]
            .expand(
                query_current.shape[0],
                attention_heads,
                prefix,
                query_current.shape[-1],
            )
            .contiguous()
        )
        q_full = torch.cat(
            [
                prefix_fill,
                query_current,
            ],
            dim=-2,
        )
    else:
        q_full = query_current

    block_map = _build_mass_mask(
        q_full,
        key_full,
    )

    current_q_blocks = math.ceil(
        q_len / QUERY_BLOCK
    )
    current_map = block_map[
        ...,
        -current_q_blocks:,
        :,
    ]
    active_blocks = int(
        current_map.sum().item()
    )
    total_blocks = int(
        current_map.numel()
    )
    sparsity = (
        1.0
        - active_blocks
        / max(total_blocks, 1)
    )

    start = torch.cuda.Event(
        enable_timing=True
    )
    end = torch.cuda.Event(
        enable_timing=True
    )

    start.record()
    output_full = _ampere_block_sparse_causal(
        q_full,
        key_full,
        value_full,
        block_map,
        scale,
    )
    end.record()
    end.synchronize()

    elapsed_ms = float(
        start.elapsed_time(end)
    )

    output_current = output_full[
        ...,
        -q_len:,
        :,
    ]

    return SparseAttentionResult(
        output=output_current,
        active_blocks=active_blocks,
        elapsed_ms=elapsed_ms,
        sparsity=float(sparsity),
    )
