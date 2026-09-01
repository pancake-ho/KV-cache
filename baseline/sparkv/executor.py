from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
)

from baseline.sparkv.codec import (
    decode_encoded_bytes,
)
from baseline.sparkv.sparge import (
    sparse_attention_current_chunk,
)
from baseline.sparkv.runtime_controller import (
    RuntimeController,
    RuntimeControllerConfig,
)
from baseline.sparkv.scheduler import Chunk


@dataclass
class ExecutionStats:
    wire_bytes: int = 0
    wire_ms: float = 0.0
    decode_ms: float = 0.0
    h2d_ms: float = 0.0

    compute_wall_ms: float = 0.0
    support_attention_ms: float = 0.0
    physical_dependency_wait_ms: float = 0.0

    runtime_migrations: int = 0
    permanently_blocked_to_stream: int = 0
    deferred_compute_events: int = 0
    forced_stream_recovery: int = 0

    local_units: int = 0
    streamed_units: int = 0

    schedule_predicted_makespan_ms: float = 0.0
    actual_context_rebuild_ms: float = 0.0

    stage_records: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "wire_bytes":
                self.wire_bytes,
            "wire_ms":
                self.wire_ms,
            "decode_ms":
                self.decode_ms,
            "h2d_ms":
                self.h2d_ms,
            "compute_wall_ms":
                self.compute_wall_ms,
            "support_attention_ms":
                self.support_attention_ms,
            "physical_dependency_wait_ms":
                self.physical_dependency_wait_ms,
            "runtime_migrations":
                self.runtime_migrations,
            "permanently_blocked_to_stream":
                self.permanently_blocked_to_stream,
            "deferred_compute_events":
                self.deferred_compute_events,
            "forced_stream_recovery":
                self.forced_stream_recovery,
            "local_units":
                self.local_units,
            "streamed_units":
                self.streamed_units,
            "schedule_predicted_makespan_ms":
                self.schedule_predicted_makespan_ms,
            "actual_context_rebuild_ms":
                self.actual_context_rebuild_ms,
            "stage_records":
                self.stage_records,
            "executor":
                "sparkv-direct-qwen3-sparge-v1",
            "stream_granularity":
                "(token,layer,kv-head)",
            "local_granularity":
                "(token,layer,kv-head) route; full-layer hidden-state "
                "finalization for physical Transformer consistency",
            "runtime_adaptation":
                True,
            "actual_huffman_bitstream":
                True,
            "spargeattention":
                "official Ampere CUDA binding, causal=True",
        }


class UnitStore:
    def __init__(
        self,
    ) -> None:
        self._values: dict[
            Chunk,
            tuple[
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        self._routes: dict[
            Chunk,
            str,
        ] = {}
        self._cond = (
            threading.Condition()
        )

    def put(
        self,
        c: Chunk,
        pair: tuple[
            torch.Tensor,
            torch.Tensor,
        ],
        route: str,
    ) -> None:
        with self._cond:
            if c in self._values:
                raise RuntimeError(
                    f"duplicate KV ownership: {c}"
                )
            self._values[c] = pair
            self._routes[c] = route
            self._cond.notify_all()

    def has(
        self,
        c: Chunk,
    ) -> bool:
        with self._cond:
            return (
                c in self._values
            )

    def get(
        self,
        c: Chunk,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        with self._cond:
            if c not in self._values:
                raise KeyError(c)
            return self._values[c]

    def route(
        self,
        c: Chunk,
    ) -> str | None:
        with self._cond:
            return self._routes.get(
                c
            )

    def count(self) -> int:
        with self._cond:
            return len(
                self._values
            )


class CloudMemorySource:
    """
    Preload cloud artifacts into host memory outside TTFT.

    This avoids accidentally measuring local filesystem latency as wireless
    transfer latency.  During the request we charge only b_c / bw plus actual
    Huffman decode and x86 host-to-device copy.
    """

    def __init__(
        self,
        sample_dir: Path,
        meta: dict[str, Any],
    ) -> None:
        self.blobs: dict[
            str,
            bytes,
        ] = {}

        for key, info in (
            meta[
                "unit_files"
            ].items()
        ):
            path = (
                sample_dir
                / info["path"]
            )
            blob = (
                path.read_bytes()
            )

            expected = int(
                info[
                    "wire_bytes"
                ]
            )
            if len(blob) != expected:
                raise ValueError(
                    "wire byte metadata mismatch for "
                    f"{key}: expected={expected}, got={len(blob)}"
                )

            self.blobs[
                key
            ] = blob

    @staticmethod
    def key(
        c: Chunk,
    ) -> str:
        return c.key

    def get(
        self,
        c: Chunk,
    ) -> bytes:
        return self.blobs[
            self.key(c)
        ]


def _effective_bandwidth(
    mean_mbps: float,
    cv: float,
    rng: np.random.Generator,
) -> float:
    if mean_mbps <= 0:
        raise ValueError(
            "bandwidth_mbps must be positive"
        )
    if cv < 0:
        raise ValueError(
            "jitter_cv must be non-negative"
        )
    if cv == 0:
        return mean_mbps

    sigma2 = math.log1p(
        cv * cv
    )
    factor = rng.lognormal(
        mean=-0.5 * sigma2,
        sigma=math.sqrt(
            sigma2
        ),
    )
    return max(
        1e-3,
        mean_mbps
        * factor,
    )


class HybridQwen3Engine:
    """
    Direct Qwen3 local-compute path for SparKV.

    KV ownership remains exactly per (t,l,h).  A local unit projects its KV
    from the current hidden state and evaluates the corresponding GQA query
    head group with SpargeAttention.  If at least one head of a non-final
    layer is locally computed, the complete Transformer hidden state is
    finalized once all head partitions required by that layer are available.

    The latter synchronization is a physical Transformer requirement that is
    not expressed in the paper's per-head readiness equations.  We therefore
    record any resulting waits instead of silently materializing unrelated
    cache units as the old executor did.
    """

    def __init__(
        self,
        *,
        model: Any,
        record: dict[str, Any],
        runtime: Any,
        chunk_size: int,
        token_chunks: int,
        layers: int,
        kv_heads: int,
        store: UnitStore,
        stats: ExecutionStats,
    ) -> None:
        self.model = model
        self.record = record
        self.runtime = runtime
        self.chunk_size = int(
            chunk_size
        )
        self.T = int(
            token_chunks
        )
        self.L = int(
            layers
        )
        self.H = int(
            kv_heads
        )
        self.store = store
        self.stats = stats

        self.hidden_inputs: dict[
            tuple[int, int],
            torch.Tensor,
        ] = {}

        self.projection_cache: dict[
            tuple[int, int],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}

        self.attention_parts: dict[
            Chunk,
            torch.Tensor,
        ] = {}

        self.local_heads: dict[
            tuple[int, int],
            set[int],
        ] = {}

        self.finalized_layers: set[
            tuple[int, int]
        ] = set()

        self._lock = (
            threading.RLock()
        )

    def _ids_for_chunk(
        self,
        t: int,
    ) -> torch.Tensor:
        start = (
            t
            * self.chunk_size
        )
        end = (
            start
            + self.chunk_size
        )
        ids = self.record[
            "prefill_ids"
        ][
            start:end
        ]
        if (
            len(ids)
            != self.chunk_size
        ):
            raise RuntimeError(
                f"invalid input chunk t={t}"
            )
        return torch.tensor(
            ids,
            dtype=torch.long,
            device=self.runtime.device,
        )[None]

    def _hidden_input(
        self,
        t: int,
        layer_idx: int,
    ) -> torch.Tensor | None:
        key = (
            t,
            layer_idx,
        )

        if key in (
            self.hidden_inputs
        ):
            return self.hidden_inputs[
                key
            ]

        if layer_idx != 0:
            return None

        ids = self._ids_for_chunk(
            t
        )
        hidden = (
            self.model.model
            .embed_tokens(
                ids
            )
        )
        self.hidden_inputs[
            key
        ] = hidden
        return hidden

    def _projection_context(
        self,
        t: int,
        layer_idx: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None:
        key = (
            t,
            layer_idx,
        )

        if key in (
            self.projection_cache
        ):
            return (
                self.projection_cache[
                    key
                ]
            )

        hidden = (
            self._hidden_input(
                t,
                layer_idx,
            )
        )
        if hidden is None:
            return None

        layer = (
            self.model.model.layers[
                layer_idx
            ]
        )
        attn = (
            layer.self_attn
        )

        normed = (
            layer.input_layernorm(
                hidden
            )
        )
        input_shape = (
            normed.shape[:-1]
        )
        head_dim = int(
            attn.head_dim
        )

        q = (
            attn.q_proj(
                normed
            )
            .view(
                *input_shape,
                -1,
                head_dim,
            )
            .transpose(
                1,
                2,
            )
        )
        k = (
            attn.k_proj(
                normed
            )
            .view(
                *input_shape,
                -1,
                head_dim,
            )
            .transpose(
                1,
                2,
            )
        )
        v = (
            attn.v_proj(
                normed
            )
            .view(
                *input_shape,
                -1,
                head_dim,
            )
            .transpose(
                1,
                2,
            )
        )

        q = (
            attn.q_norm(
                q.transpose(
                    1,
                    2,
                )
            )
            .transpose(
                1,
                2,
            )
        )
        k = (
            attn.k_norm(
                k.transpose(
                    1,
                    2,
                )
            )
            .transpose(
                1,
                2,
            )
        )

        start = (
            t
            * self.chunk_size
        )
        position_ids = (
            torch.arange(
                start,
                start
                + self.chunk_size,
                dtype=torch.long,
                device=
                    self.runtime.device,
            )[None]
        )

        position_embeddings = (
            self.model.model
            .rotary_emb(
                hidden,
                position_ids,
            )
        )
        cos, sin = (
            position_embeddings
        )

        q, k = (
            apply_rotary_pos_emb(
                q,
                k,
                cos,
                sin,
            )
        )

        value = (
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
        )
        self.projection_cache[
            key
        ] = value
        return value

    def _history_available(
        self,
        c: Chunk,
    ) -> bool:
        return all(
            self.store.has(
                Chunk(
                    previous_t,
                    c.layer,
                    c.head,
                )
            )
            for previous_t in range(
                c.t
            )
        )

    def _history_with_current(
        self,
        c: Chunk,
        current_pair: tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        keys = []
        values = []

        for previous_t in range(
            c.t
        ):
            key, value = (
                self.store.get(
                    Chunk(
                        previous_t,
                        c.layer,
                        c.head,
                    )
                )
            )
            keys.append(key)
            values.append(
                value
            )

        keys.append(
            current_pair[0]
        )
        values.append(
            current_pair[1]
        )

        return (
            torch.cat(
                keys,
                dim=-2,
            )
            if len(keys) > 1
            else keys[0],
            torch.cat(
                values,
                dim=-2,
            )
            if len(values) > 1
            else values[0],
        )

    def _compute_attention_part(
        self,
        c: Chunk,
        *,
        current_pair: tuple[
            torch.Tensor,
            torch.Tensor,
        ],
        support: bool,
    ) -> bool:
        if (
            c.layer
            == self.L - 1
        ):
            return True

        if not (
            self._history_available(
                c
            )
        ):
            return False

        context = (
            self._projection_context(
                c.t,
                c.layer,
            )
        )
        if context is None:
            return False

        q, _, _ = context
        layer = (
            self.model.model.layers[
                c.layer
            ]
        )
        groups = int(
            layer.self_attn
            .num_key_value_groups
        )

        q_start = (
            c.head
            * groups
        )
        q_end = (
            q_start
            + groups
        )

        key_prefix, value_prefix = (
            self._history_with_current(
                c,
                current_pair,
            )
        )

        begin = (
            time.perf_counter()
        )

        result = (
            sparse_attention_current_chunk(
                query_current=
                    q[
                        :,
                        q_start:q_end,
                        :,
                        :,
                    ].contiguous(),
                key_history_and_current=
                    key_prefix,
                value_history_and_current=
                    value_prefix,
                current_chunk_tokens=
                    self.chunk_size,
                num_key_value_groups=
                    groups,
                scale=float(
                    layer.self_attn.scaling
                ),
            )
        )

        wall_ms = (
            time.perf_counter()
            - begin
        ) * 1000.0

        self.attention_parts[
            c
        ] = (
            result.output
        )

        if support:
            self.stats.support_attention_ms += (
                wall_ms
            )

        return True

    def compute_unit(
        self,
        c: Chunk,
    ) -> bool:
        with self._lock:
            if self.store.has(c):
                return True

            context = (
                self._projection_context(
                    c.t,
                    c.layer,
                )
            )
            if context is None:
                return False

            if (
                c.layer
                < self.L - 1
                and not
                self._history_available(
                    c
                )
            ):
                return False

            _, k_current, v_current = (
                context
            )

            local_pair = (
                k_current[
                    :,
                    c.head:
                    c.head + 1,
                    :,
                    :,
                ].contiguous(),
                v_current[
                    :,
                    c.head:
                    c.head + 1,
                    :,
                    :,
                ].contiguous(),
            )

            # Non-final t_comp includes sparse attention + dense layer work.
            # Final layer is projection-only in the paper.
            if (
                c.layer
                < self.L - 1
            ):
                if not (
                    self._compute_attention_part(
                        c,
                        current_pair=
                            local_pair,
                        support=False,
                    )
                ):
                    return False

            self.store.put(
                c,
                local_pair,
                "compute",
            )

            self.local_heads.setdefault(
                (
                    c.t,
                    c.layer,
                ),
                set(),
            ).add(
                c.head
            )

            self.stats.local_units += 1

            return True

    def _all_layer_units_available(
        self,
        t: int,
        layer_idx: int,
    ) -> bool:
        return all(
            self.store.has(
                Chunk(
                    token,
                    layer_idx,
                    head,
                )
            )
            for token in range(
                t + 1
            )
            for head in range(
                self.H
            )
        )

    def try_finalize_layer(
        self,
        t: int,
        layer_idx: int,
    ) -> bool:
        with self._lock:
            key = (
                t,
                layer_idx,
            )

            if key in (
                self.finalized_layers
            ):
                return False

            if (
                layer_idx
                >= self.L - 1
            ):
                return False

            # If no head at this (t,l) is locally computed, there is no
            # upward local-compute dependency and no need to execute the
            # Transformer layer solely for hidden-state continuation.
            if not (
                self.local_heads.get(
                    key
                )
            ):
                return False

            if not (
                self._all_layer_units_available(
                    t,
                    layer_idx,
                )
            ):
                return False

            context = (
                self._projection_context(
                    t,
                    layer_idx,
                )
            )
            if context is None:
                return False

            # Complete attention outputs for streamed heads as support work
            # needed to form the physically valid shared hidden state.
            for head in range(
                self.H
            ):
                c = Chunk(
                    t,
                    layer_idx,
                    head,
                )

                if c in (
                    self.attention_parts
                ):
                    continue

                pair = (
                    self.store.get(
                        c
                    )
                )

                if not (
                    self._compute_attention_part(
                        c,
                        current_pair=
                            pair,
                        support=True,
                    )
                ):
                    return False

            layer = (
                self.model.model.layers[
                    layer_idx
                ]
            )
            hidden = (
                self._hidden_input(
                    t,
                    layer_idx,
                )
            )
            assert (
                hidden is not None
            )

            parts = [
                self.attention_parts[
                    Chunk(
                        t,
                        layer_idx,
                        head,
                    )
                ]
                for head in range(
                    self.H
                )
            ]

            attn_heads = torch.cat(
                parts,
                dim=1,
            )
            attn_out = (
                attn_heads
                .transpose(
                    1,
                    2,
                )
                .contiguous()
                .reshape(
                    hidden.shape[
                        0
                    ],
                    hidden.shape[
                        1
                    ],
                    -1,
                )
            )
            attn_out = (
                layer.self_attn
                .o_proj(
                    attn_out
                )
            )

            after_attn = (
                hidden
                + attn_out
            )
            residual = (
                after_attn
            )
            ffn_input = (
                layer
                .post_attention_layernorm(
                    after_attn
                )
            )
            output = (
                residual
                + layer.mlp(
                    ffn_input
                )
            )

            self.hidden_inputs[
                (
                    t,
                    layer_idx
                    + 1,
                )
            ] = output
            self.finalized_layers.add(
                key
            )

            # Projection/attention state at this layer is no longer required
            # once the hidden state for the next layer has been materialized.
            # Keep attention_parts only until request completion for simpler
            # debugging; clear the large shared projections.
            self.projection_cache.pop(
                key,
                None,
            )

            return True

    def finalize_all_possible(
        self,
    ) -> int:
        progress = 0
        changed = True

        while changed:
            changed = False

            for layer_idx in range(
                self.L - 1
            ):
                for t in range(
                    self.T
                ):
                    if (
                        self.try_finalize_layer(
                            t,
                            layer_idx,
                        )
                    ):
                        changed = True
                        progress += 1

        return progress

    def assemble_cache(
        self,
    ) -> DynamicCache:
        if (
            self.store.count()
            != self.T
            * self.L
            * self.H
        ):
            raise RuntimeError(
                "cannot assemble incomplete SparKV cache: "
                f"have={self.store.count()}, "
                f"expected={self.T*self.L*self.H}"
            )

        legacy = []

        for layer_idx in range(
            self.L
        ):
            head_keys = []
            head_values = []

            for head in range(
                self.H
            ):
                token_keys = []
                token_values = []

                for t in range(
                    self.T
                ):
                    key, value = (
                        self.store.get(
                            Chunk(
                                t,
                                layer_idx,
                                head,
                            )
                        )
                    )
                    token_keys.append(
                        key
                    )
                    token_values.append(
                        value
                    )

                head_keys.append(
                    torch.cat(
                        token_keys,
                        dim=-2,
                    )
                )
                head_values.append(
                    torch.cat(
                        token_values,
                        dim=-2,
                    )
                )

            legacy.append(
                (
                    torch.cat(
                        head_keys,
                        dim=1,
                    ),
                    torch.cat(
                        head_values,
                        dim=1,
                    ),
                )
            )

        cache = (
            DynamicCache
            .from_legacy_cache(
                tuple(
                    legacy
                )
            )
        )

        expected = (
            self.T
            * self.chunk_size
        )
        if (
            int(
                cache
                .get_seq_length()
            )
            != expected
        ):
            raise RuntimeError(
                "assembled cache length mismatch: "
                f"expected={expected}, "
                f"got={cache.get_seq_length()}"
            )

        return cache


def _load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _item_chunk(
    item: dict[str, Any],
) -> Chunk:
    return Chunk(
        int(item["t"]),
        int(item["layer"]),
        int(item["head"]),
    )


def _permanently_layer_blocked(
    c: Chunk,
    done: dict[
        Chunk,
        str,
    ],
) -> bool:
    if c.layer == 0:
        return False

    previous = Chunk(
        c.t,
        c.layer - 1,
        c.head,
    )

    return (
        previous in done
        and done[
            previous
        ]
        != "compute"
    )


def execute_sparkv(
    *,
    model: Any,
    record: dict[str, Any],
    runtime: Any,
    sample_dir: Path,
    schedule_path: Path,
    bandwidth_mbps: float,
    jitter_cv: float,
    seed: int,
    controller_config: RuntimeControllerConfig,
) -> tuple[
    DynamicCache,
    ExecutionStats,
]:
    if not runtime.is_cuda:
        raise RuntimeError(
            "Direct SparKV executor requires CUDA"
        )

    meta = _load_json(
        sample_dir
        / "meta.json"
    )
    schedule = _load_json(
        schedule_path
    )

    T = int(
        meta[
            "token_chunks"
        ]
    )
    L = int(
        meta["layers"]
    )
    H = int(
        meta[
            "kv_heads"
        ]
    )
    chunk_size = int(
        meta[
            "chunk_size"
        ]
    )

    if (
        len(
            record[
                "prefill_ids"
            ]
        )
        != T
        * chunk_size
    ):
        raise ValueError(
            "record/cache geometry mismatch"
        )

    if int(
        schedule[
            "chunks"
        ]
    ) != (
        T
        * L
        * H
    ):
        raise ValueError(
            "schedule/cache unit count mismatch"
        )

    store = UnitStore()
    stats = ExecutionStats(
        schedule_predicted_makespan_ms=
            float(
                schedule[
                    "makespan_ms"
                ]
            )
    )

    # Intentionally outside request timing.
    cloud = (
        CloudMemorySource(
            sample_dir,
            meta,
        )
    )

    engine = HybridQwen3Engine(
        model=model,
        record=record,
        runtime=runtime,
        chunk_size=
            chunk_size,
        token_chunks=T,
        layers=L,
        kv_heads=H,
        store=store,
        stats=stats,
    )

    controller = RuntimeController(
        layers=L,
        config=
            controller_config,
    )

    stages = [
        dict(stage)
        for stage in
        schedule[
            "stages"
        ]
    ]
    unit_costs = schedule[
        "unit_costs"
    ]

    rng = (
        np.random.default_rng(
            seed
        )
    )

    done: dict[
        Chunk,
        str,
    ] = {}
    done_lock = (
        threading.Lock()
    )
    pending_compute: list[
        dict[str, Any]
    ] = []

    copy_stream = (
        torch.cuda.Stream(
            device=
                runtime.device
        )
    )

    rebuild_begin = (
        time.perf_counter()
    )

    stage_idx = 0
    tail_stall_count = 0
    while (
        stage_idx
        < len(stages)
        or pending_compute
    ):
        if stage_idx < len(
            stages
        ):
            current = dict(
                stages[
                    stage_idx
                ]
            )
            next_stage = (
                dict(
                    stages[
                        stage_idx
                        + 1
                    ]
                )
                if (
                    stage_idx
                    + 1
                    < len(stages)
                )
                else None
            )

            with done_lock:
                done_snapshot = dict(
                    done
                )

            (
                current,
                adapted_next,
                adaptation,
            ) = controller.adapt(
                current_stage=
                    current,
                next_stage=
                    next_stage,
                done=
                    done_snapshot,
                unit_costs=
                    unit_costs,
            )

            if (
                adapted_next
                is not None
            ):
                stages[
                    stage_idx
                    + 1
                ] = (
                    adapted_next
                )

            stats.runtime_migrations += int(
                adaptation.get(
                    "migrations",
                    0,
                )
            )

            compute_items = (
                pending_compute
                + list(
                    current.get(
                        "compute",
                        [],
                    )
                )
            )
            pending_compute = []

            stream_items = list(
                current.get(
                    "stream",
                    [],
                )
            )
            stage_label = int(
                current.get(
                    "stage",
                    stage_idx + 1,
                )
            )
        else:
            # Tail stage to finish physically deferred compute operations.
            compute_items = (
                pending_compute
            )
            pending_compute = []
            stream_items = []
            adaptation = {
                "events": [],
                "migrations": 0,
                "tail": True,
            }
            stage_label = (
                len(stages) + 1
            )

        # A runtime compute->stream migration can permanently invalidate
        # downstream local-compute dependencies.  Convert those dependent
        # compute chunks to stream instead of attempting impossible compute.
        still_compute = []
        for item in compute_items:
            c = _item_chunk(
                item
            )
            with done_lock:
                blocked = (
                    _permanently_layer_blocked(
                        c,
                        done,
                    )
                )
            if blocked:
                stream_items.append(
                    item
                )
                stats.permanently_blocked_to_stream += 1
            else:
                still_compute.append(
                    item
                )
        compute_items = (
            still_compute
        )

        predicted_compute_ms = sum(
            float(
                unit_costs[
                    _item_chunk(
                        item
                    ).key
                ][
                    "comp_ms"
                ]
            )
            for item in
            compute_items
        )
        predicted_stream_ms = sum(
            float(
                unit_costs[
                    _item_chunk(
                        item
                    ).key
                ][
                    "stream_ms"
                ]
            )
            for item in
            stream_items
        )

        stream_finished = (
            threading.Event()
        )
        stream_error: list[
            BaseException
        ] = []
        compute_error: list[
            BaseException
        ] = []
        deferred: list[
            dict[str, Any]
        ] = []

        stage_stream_begin = [
            None
        ]
        stage_stream_end = [
            None
        ]
        stage_compute_begin = [
            None
        ]
        stage_compute_end = [
            None
        ]

        def stream_worker() -> None:
            try:
                torch.cuda.set_device(
                    runtime.device
                )
                stage_stream_begin[
                    0
                ] = time.perf_counter()

                for item in (
                    stream_items
                ):
                    c = _item_chunk(
                        item
                    )

                    with done_lock:
                        if c in done:
                            continue

                    blob = cloud.get(
                        c
                    )
                    bw = (
                        _effective_bandwidth(
                            bandwidth_mbps,
                            jitter_cv,
                            rng,
                        )
                    )
                    wire_seconds = (
                        len(blob)
                        * 8.0
                        / (
                            bw
                            * 1e6
                        )
                    )
                    time.sleep(
                        wire_seconds
                    )

                    decode_begin = (
                        time.perf_counter()
                    )
                    key_cpu, value_cpu, _ = (
                        decode_encoded_bytes(
                            blob,
                            dtype=
                                runtime.dtype,
                        )
                    )
                    stats.decode_ms += (
                        time.perf_counter()
                        - decode_begin
                    ) * 1000.0

                    try:
                        key_cpu = (
                            key_cpu
                            .pin_memory()
                        )
                        value_cpu = (
                            value_cpu
                            .pin_memory()
                        )
                    except RuntimeError:
                        pass

                    h2d_begin = (
                        time.perf_counter()
                    )
                    with torch.cuda.stream(
                        copy_stream
                    ):
                        key_gpu = (
                            key_cpu.to(
                                runtime.device,
                                non_blocking=True,
                            )
                        )
                        value_gpu = (
                            value_cpu.to(
                                runtime.device,
                                non_blocking=True,
                            )
                        )
                    copy_stream.synchronize()
                    stats.h2d_ms += (
                        time.perf_counter()
                        - h2d_begin
                    ) * 1000.0

                    store.put(
                        c,
                        (
                            key_gpu,
                            value_gpu,
                        ),
                        "stream",
                    )

                    with done_lock:
                        done[c] = (
                            "stream"
                        )

                    stats.wire_bytes += (
                        len(blob)
                    )
                    stats.wire_ms += (
                        wire_seconds
                        * 1000.0
                    )
                    stats.streamed_units += 1

                stage_stream_end[
                    0
                ] = time.perf_counter()

            except BaseException as exc:
                stream_error.append(
                    exc
                )
            finally:
                stream_finished.set()

        def compute_worker() -> None:
            try:
                torch.cuda.set_device(
                    runtime.device
                )
                stage_compute_begin[
                    0
                ] = time.perf_counter()

                remaining = list(
                    compute_items
                )

                idle_begin = None

                while remaining:
                    progress = False
                    next_remaining = []

                    for item in (
                        remaining
                    ):
                        c = (
                            _item_chunk(
                                item
                            )
                        )

                        with done_lock:
                            if c in done:
                                continue

                            # Re-check paper readiness under any runtime
                            # migrations that have already completed.
                            ready = (
                                controller
                                .compute_ready(
                                    c,
                                    done,
                                )
                            )
                            blocked = (
                                _permanently_layer_blocked(
                                    c,
                                    done,
                                )
                            )

                        if blocked:
                            # Cannot mutate stream_items after the stream
                            # worker has started.  Defer; the next stage will
                            # convert it to streaming before threads launch.
                            next_remaining.append(
                                item
                            )
                            continue

                        if not ready:
                            next_remaining.append(
                                item
                            )
                            continue

                        if not (
                            engine.compute_unit(
                                c
                            )
                        ):
                            next_remaining.append(
                                item
                            )
                            continue

                        with done_lock:
                            done[c] = (
                                "compute"
                            )

                        progress = True
                        engine.finalize_all_possible()

                    remaining = (
                        next_remaining
                    )

                    if not remaining:
                        break

                    if progress:
                        idle_begin = None
                        continue

                    # Let the concurrent streaming path deliver physical
                    # dependencies.  If it has finished and no further
                    # progress is possible, carry the compute operations to
                    # the next stage.
                    if stream_finished.is_set():
                        break

                    if idle_begin is None:
                        idle_begin = (
                            time.perf_counter()
                        )

                    time.sleep(
                        0.001
                    )

                if idle_begin is not None:
                    stats.physical_dependency_wait_ms += (
                        time.perf_counter()
                        - idle_begin
                    ) * 1000.0

                deferred.extend(
                    remaining
                )
                if remaining:
                    stats.deferred_compute_events += (
                        len(
                            remaining
                        )
                    )

                stage_compute_end[
                    0
                ] = time.perf_counter()

            except BaseException as exc:
                compute_error.append(
                    exc
                )

        stream_thread = (
            threading.Thread(
                target=
                    stream_worker,
                daemon=True,
                name=
                    "sparkv-stream",
            )
        )
        compute_thread = (
            threading.Thread(
                target=
                    compute_worker,
                daemon=True,
                name=
                    "sparkv-compute",
            )
        )

        units_before_stage = store.count()

        stage_wall_begin = (
            time.perf_counter()
        )

        stream_thread.start()
        compute_thread.start()

        stream_thread.join()
        compute_thread.join()

        if stream_error:
            raise stream_error[0]
        if compute_error:
            raise compute_error[0]

        engine.finalize_all_possible()

        stage_wall_ms = (
            time.perf_counter()
            - stage_wall_begin
        ) * 1000.0

        actual_stream_ms = (
            0.0
            if (
                stage_stream_begin[
                    0
                ]
                is None
                or stage_stream_end[
                    0
                ]
                is None
            )
            else (
                stage_stream_end[
                    0
                ]
                - stage_stream_begin[
                    0
                ]
            )
            * 1000.0
        )

        actual_compute_ms = (
            0.0
            if (
                stage_compute_begin[
                    0
                ]
                is None
                or stage_compute_end[
                    0
                ]
                is None
            )
            else (
                stage_compute_end[
                    0
                ]
                - stage_compute_begin[
                    0
                ]
            )
            * 1000.0
        )

        stats.compute_wall_ms += (
            actual_compute_ms
        )

        controller.observe_stage(
            predicted_compute_ms=
                predicted_compute_ms,
            actual_compute_ms=
                actual_compute_ms,
            predicted_stream_ms=
                predicted_stream_ms,
            actual_stream_ms=
                actual_stream_ms,
        )

        stats.stage_records.append(
            {
                "stage":
                    stage_label,
                "predicted_compute_ms":
                    predicted_compute_ms,
                "actual_compute_ms":
                    actual_compute_ms,
                "predicted_stream_ms":
                    predicted_stream_ms,
                "actual_stream_ms":
                    actual_stream_ms,
                "stage_wall_ms":
                    stage_wall_ms,
                "deferred_compute":
                    len(
                        deferred
                    ),
                "adaptation":
                    adaptation,
                "compute_ratio_window":
                    controller.compute_ratio,
                "stream_ratio_window":
                    controller.stream_ratio,
            }
        )

        pending_compute.extend(
            deferred
        )

        if stage_idx < len(
            stages
        ):
            stage_idx += 1
        else:
            # Tail recovery.  A valid paper schedule should normally finish
            # without entering this branch.  If physical shared-hidden-state
            # constraints prevent further local progress, explicitly migrate
            # the remaining units to streaming rather than looping forever.
            if pending_compute:
                engine.finalize_all_possible()

                units_after_stage = store.count()
                if units_after_stage == units_before_stage:
                    tail_stall_count += 1
                else:
                    tail_stall_count = 0

                with done_lock:
                    snapshot = dict(
                        done
                    )

                recoverable = []
                still = []

                for item in pending_compute:
                    c = _item_chunk(item)
                    if (
                        _permanently_layer_blocked(
                            c,
                            snapshot,
                        )
                        or not controller.compute_ready(
                            c,
                            snapshot,
                        )
                        or tail_stall_count >= 1
                    ):
                        recoverable.append(item)
                    else:
                        still.append(item)

                if recoverable:
                    stages.append(
                        {
                            "stage":
                                len(stages) + 1,
                            "compute":
                                still,
                            "stream":
                                recoverable,
                        }
                    )
                    pending_compute = []
                    stats.forced_stream_recovery += len(
                        recoverable
                    )
                else:
                    pending_compute = still

    engine.finalize_all_possible()

    expected_units = (
        T * L * H
    )
    if store.count() != expected_units:
        missing = [
            Chunk(
                t,
                layer,
                head,
            )
            for t in range(T)
            for layer in range(L)
            for head in range(H)
            if not store.has(
                Chunk(
                    t,
                    layer,
                    head,
                )
            )
        ]
        raise RuntimeError(
            "SparKV execution ended with incomplete cache. "
            f"missing={missing[:10]}"
        )

    cache = (
        engine.assemble_cache()
    )

    stats.actual_context_rebuild_ms = (
        time.perf_counter()
        - rebuild_begin
    ) * 1000.0

    return (
        cache,
        stats,
    )
