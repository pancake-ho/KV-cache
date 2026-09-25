from __future__ import annotations

import re
from typing import Any

import torch


_LAYER_QNORM_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)"
    r"\.self_attn\.q_norm$"
)

_LAYER_QPROJ_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)"
    r"\.self_attn\.q_proj$"
)


def _as_tensor(
    output: Any,
) -> torch.Tensor:
    if isinstance(
        output,
        torch.Tensor,
    ):
        return output

    if (
        isinstance(
            output,
            (list, tuple),
        )
        and output
        and isinstance(
            output[0],
            torch.Tensor,
        )
    ):
        return output[0]

    raise TypeError(
        "Unsupported hook output type: "
        f"{type(output)}"
    )


def _normalize_q(
    tensor: torch.Tensor,
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    mode: str,
) -> torch.Tensor:
    """
    Return Q as [T, Hq, Dh].
    """
    tensor = tensor.detach()

    if mode == "q_norm":
        if tensor.ndim != 4:
            raise RuntimeError(
                "q_norm output must be rank-4, "
                f"got {list(tensor.shape)}"
            )

        if tensor.shape[0] != 1:
            raise RuntimeError(
                "Only batch size 1 is supported."
            )

        x = tensor[0]

        # [T, H, D]
        if (
            x.shape[0] == seq_len
            and x.shape[1] == num_heads
            and x.shape[2] == head_dim
        ):
            return x

        # [H, T, D]
        if (
            x.shape[0] == num_heads
            and x.shape[1] == seq_len
            and x.shape[2] == head_dim
        ):
            return x.permute(
                1, 0, 2
            )

        raise RuntimeError(
            "Unexpected q_norm output shape: "
            f"{list(tensor.shape)}"
        )

    if mode == "q_proj":
        if tensor.ndim != 3:
            raise RuntimeError(
                "q_proj output must be rank-3, "
                f"got {list(tensor.shape)}"
            )

        if tensor.shape[0] != 1:
            raise RuntimeError(
                "Only batch size 1 is supported."
            )

        x = tensor[0]

        expected_width = (
            num_heads * head_dim
        )

        if (
            x.shape[0] != seq_len
            or x.shape[1]
            != expected_width
        ):
            raise RuntimeError(
                "Unexpected q_proj shape: "
                f"{list(tensor.shape)}"
            )

        return x.reshape(
            seq_len,
            num_heads,
            head_dim,
        )

    raise ValueError(
        f"Unsupported Q mode={mode}"
    )


def _normalize_cache_tensor(
    tensor: torch.Tensor,
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Return cache tensor as [T, Hkv, Dh].
    """
    if tensor.ndim != 4:
        raise RuntimeError(
            "KV cache must be rank-4, "
            f"got {list(tensor.shape)}"
        )

    if tensor.shape[0] != 1:
        raise RuntimeError(
            "Only batch size 1 is supported."
        )

    x = tensor[0]

    # HF cache standard: [H, T, D]
    if (
        x.shape[0] == num_heads
        and x.shape[1] == seq_len
        and x.shape[2] == head_dim
    ):
        return x.permute(
            1, 0, 2
        )

    # Defensive fallback: [T, H, D]
    if (
        x.shape[0] == seq_len
        and x.shape[1] == num_heads
        and x.shape[2] == head_dim
    ):
        return x

    raise RuntimeError(
        "Unexpected KV cache shape: "
        f"{list(tensor.shape)}"
    )


def extract_cache_lists(
    past_key_values,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """
    Support both DynamicCache and legacy cache representations.
    """
    if (
        hasattr(
            past_key_values,
            "key_cache",
        )
        and hasattr(
            past_key_values,
            "value_cache",
        )
    ):
        return (
            list(
                past_key_values.key_cache
            ),
            list(
                past_key_values.value_cache
            ),
        )

    if hasattr(
        past_key_values,
        "layers",
    ):
        keys = []
        values = []

        for layer in (
            past_key_values.layers
        ):
            key = getattr(
                layer,
                "keys",
                None,
            )

            value = getattr(
                layer,
                "values",
                None,
            )

            if key is None:
                key = getattr(
                    layer,
                    "key_cache",
                    None,
                )

            if value is None:
                value = getattr(
                    layer,
                    "value_cache",
                    None,
                )

            if (
                key is None
                or value is None
            ):
                raise RuntimeError(
                    "Cannot locate K/V in "
                    "Cache layer"
                )

            keys.append(key)
            values.append(value)

        return keys, values

    if isinstance(
        past_key_values,
        (tuple, list),
    ):
        keys = []
        values = []

        for item in past_key_values:
            keys.append(item[0])
            values.append(item[1])

        return keys, values

    raise TypeError(
        "Unsupported cache type: "
        f"{type(past_key_values)}"
    )


class PrefillQKVCapture:

    def __init__(
        self,
        model,
        *,
        num_layers: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        self.model = model

        self.num_layers = int(
            num_layers
        )

        self.num_q_heads = int(
            num_q_heads
        )

        self.num_kv_heads = int(
            num_kv_heads
        )

        self.head_dim = int(
            head_dim
        )

        self._handles = []
        self._q_outputs = {}

        q_norm_modules = {}

        for name, module in (
            model.named_modules()
        ):
            match = (
                _LAYER_QNORM_RE.search(
                    name
                )
            )

            if match:
                layer_idx = int(
                    match.group(1)
                )

                q_norm_modules[
                    layer_idx
                ] = module

        if (
            len(q_norm_modules)
            == self.num_layers
        ):
            self.q_mode = "q_norm"
            q_modules = q_norm_modules

        else:
            q_proj_modules = {}

            for name, module in (
                model.named_modules()
            ):
                match = (
                    _LAYER_QPROJ_RE.search(
                        name
                    )
                )

                if match:
                    layer_idx = int(
                        match.group(1)
                    )

                    q_proj_modules[
                        layer_idx
                    ] = module

            if (
                len(q_proj_modules)
                != self.num_layers
            ):
                raise RuntimeError(
                    "Could not find exactly "
                    f"{self.num_layers} q_norm "
                    "or q_proj modules. "
                    f"q_norm={len(q_norm_modules)}, "
                    f"q_proj={len(q_proj_modules)}"
                )

            self.q_mode = "q_proj"
            q_modules = q_proj_modules

        for layer_idx in range(
            self.num_layers
        ):
            module = q_modules[
                layer_idx
            ]

            handle = (
                module.register_forward_hook(
                    self._make_q_hook(
                        layer_idx
                    )
                )
            )

            self._handles.append(
                handle
            )

    def _make_q_hook(
        self,
        layer_idx: int,
    ):
        def hook(
            module,
            inputs,
            output,
        ):
            del module, inputs

            self._q_outputs[
                layer_idx
            ] = _as_tensor(
                output
            ).detach()

        return hook

    def clear(
        self,
    ):
        self._q_outputs.clear()

    def close(
        self,
    ):
        for handle in self._handles:
            handle.remove()

        self._handles.clear()

    def capture(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[
        str,
        list[torch.Tensor],
    ]:
        self.clear()

        seq_len = int(
            input_ids.shape[1]
        )

        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )

        if (
            output.past_key_values
            is None
        ):
            raise RuntimeError(
                "past_key_values is None"
            )

        if (
            len(self._q_outputs)
            != self.num_layers
        ):
            raise RuntimeError(
                "Q hook layer count mismatch: "
                f"{len(self._q_outputs)} "
                f"!= {self.num_layers}"
            )

        keys, values = (
            extract_cache_lists(
                output.past_key_values
            )
        )

        if (
            len(keys)
            != self.num_layers
            or len(values)
            != self.num_layers
        ):
            raise RuntimeError(
                "KV cache layer count mismatch."
            )

        q_attn = []
        k_cache = []
        v_cache = []

        for layer_idx in range(
            self.num_layers
        ):
            q = _normalize_q(
                self._q_outputs[
                    layer_idx
                ],
                seq_len=seq_len,
                num_heads=(
                    self.num_q_heads
                ),
                head_dim=(
                    self.head_dim
                ),
                mode=self.q_mode,
            )

            k = _normalize_cache_tensor(
                keys[layer_idx],
                seq_len=seq_len,
                num_heads=(
                    self.num_kv_heads
                ),
                head_dim=(
                    self.head_dim
                ),
            )

            v = _normalize_cache_tensor(
                values[layer_idx],
                seq_len=seq_len,
                num_heads=(
                    self.num_kv_heads
                ),
                head_dim=(
                    self.head_dim
                ),
            )

            if (
                not torch.isfinite(q).all()
                or not torch.isfinite(k).all()
                or not torch.isfinite(v).all()
            ):
                raise RuntimeError(
                    "Non-finite Q/K/V at "
                    f"layer {layer_idx}"
                )

            # No raw tensors are persisted to disk.
            # Keep only one prompt's tensors in host memory.
            q_attn.append(
                q.to(
                    device="cpu",
                    dtype=torch.float16,
                )
            )

            k_cache.append(
                k.to(
                    device="cpu",
                    dtype=torch.float16,
                )
            )

            v_cache.append(
                v.to(
                    device="cpu",
                    dtype=torch.float16,
                )
            )

        del output
        self.clear()

        return {
            "q_attn": q_attn,
            "k_cache": k_cache,
            "v_cache": v_cache,
        }