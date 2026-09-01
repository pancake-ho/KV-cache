from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, order=True)
class Chunk:
    """
    SparKV chunk c=(t,l,h).

    Python indices are zero-based:
      t: token-chunk index
      l: Transformer layer index
      h: KV / attention-head partition index
    """
    t: int
    layer: int
    head: int

    @property
    def key(self) -> str:
        return (
            f"{self.t}:"
            f"{self.layer}:"
            f"{self.head}"
        )


class SparKVScheduler:
    """
    Direct implementation of Section IV-B's potential-aware greedy heuristic.

    The scheduler keeps:
      Qc: compute-ready and unfinished chunks
      Qs: all unfinished chunks

    Per stage:
      1) sort Qc by wc and greedily compute under Delta t, re-evaluating
         readiness / priority after every selected compute;
      2) reset the budget to Delta t;
      3) sort Qs by ws and greedily stream under the independent stream budget.

    A chunk is processed exactly once.  Computing removes it from Qs.
    """

    def __init__(
        self,
        token_chunks: int,
        layers: int,
        heads: int,
        comp_ms: dict[Chunk, float],
        stream_ms: dict[Chunk, float],
        delta_ms: float,
    ) -> None:
        if (
            token_chunks <= 0
            or layers <= 0
            or heads <= 0
        ):
            raise ValueError(
                "invalid scheduler geometry"
            )
        if delta_ms <= 0:
            raise ValueError(
                "delta_ms must be positive"
            )

        self.T = int(
            token_chunks
        )
        self.L = int(layers)
        self.H = int(heads)
        self.delta_ms = float(
            delta_ms
        )

        self.all_chunks = {
            Chunk(
                t,
                layer,
                head,
            )
            for t in range(
                self.T
            )
            for layer in range(
                self.L
            )
            for head in range(
                self.H
            )
        }

        self.comp_ms = dict(
            comp_ms
        )
        self.stream_ms = dict(
            stream_ms
        )

        if (
            set(self.comp_ms)
            != self.all_chunks
        ):
            raise ValueError(
                "comp_ms geometry mismatch"
            )
        if (
            set(self.stream_ms)
            != self.all_chunks
        ):
            raise ValueError(
                "stream_ms geometry mismatch"
            )
        if any(
            value <= 0
            for value in
            self.comp_ms.values()
        ):
            raise ValueError(
                "compute costs must be positive"
            )
        if any(
            value <= 0
            for value in
            self.stream_ms.values()
        ):
            raise ValueError(
                "stream costs must be positive"
            )

        self.done: dict[
            Chunk,
            str,
        ] = {}

    def reset(self) -> None:
        self.done.clear()

    def token_dependency_ready(
        self,
        c: Chunk,
        done: dict[
            Chunk,
            str,
        ],
    ) -> bool:
        # Eq. (4): t=1 or l=L are boundary cases.
        if (
            c.t == 0
            or c.layer
            == self.L - 1
        ):
            return True

        previous = Chunk(
            c.t - 1,
            c.layer,
            c.head,
        )
        return previous in done

    def layer_dependency_ready(
        self,
        c: Chunk,
        done: dict[
            Chunk,
            str,
        ],
    ) -> bool:
        # Eq. (5): l=1 has no lower-layer dependency.
        if c.layer == 0:
            return True

        previous_layer = Chunk(
            c.t,
            c.layer - 1,
            c.head,
        )
        return (
            done.get(
                previous_layer
            )
            == "compute"
        )

    def compute_ready(
        self,
        c: Chunk,
        done: dict[
            Chunk,
            str,
        ]
        | None = None,
    ) -> bool:
        state = (
            self.done
            if done is None
            else done
        )
        return (
            c not in state
            and self.token_dependency_ready(
                c,
                state,
            )
            and self.layer_dependency_ready(
                c,
                state,
            )
        )

    def _candidate_children(
        self,
        c: Chunk,
    ) -> set[Chunk]:
        children: set[
            Chunk
        ] = set()

        # Horizontal dependency for the next token chunk.
        if (
            c.t + 1 < self.T
            and c.layer
            < self.L - 1
        ):
            children.add(
                Chunk(
                    c.t + 1,
                    c.layer,
                    c.head,
                )
            )

        # Vertical dependency for the next layer.
        if (
            c.layer + 1
            < self.L
        ):
            children.add(
                Chunk(
                    c.t,
                    c.layer + 1,
                    c.head,
                )
            )

        return children

    def newly_compute_ready(
        self,
        c: Chunk,
        operation: str,
    ) -> set[Chunk]:
        if operation not in {
            "compute",
            "stream",
        }:
            raise ValueError(
                f"invalid operation: {operation}"
            )

        before = {
            x
            for x in
            self._candidate_children(
                c
            )
            if self.compute_ready(
                x,
                self.done,
            )
        }

        after_state = dict(
            self.done
        )
        after_state[c] = operation

        after = {
            x
            for x in
            self._candidate_children(
                c
            )
            if self.compute_ready(
                x,
                after_state,
            )
        }

        return after - before

    def stream_priority(
        self,
        c: Chunk,
    ) -> float:
        unlocked = (
            self.newly_compute_ready(
                c,
                "stream",
            )
        )
        return (
            1.0
            / self.stream_ms[c]
            + sum(
                1.0
                / self.comp_ms[x]
                for x in unlocked
            )
        )

    def compute_priority(
        self,
        c: Chunk,
    ) -> float:
        if not self.compute_ready(
            c
        ):
            return float("-inf")

        unlocked = (
            self.newly_compute_ready(
                c,
                "compute",
            )
        )
        return (
            1.0
            / self.comp_ms[c]
            + sum(
                1.0
                / self.comp_ms[x]
                for x in unlocked
            )
        )

    def _compute_phase(
        self,
    ) -> tuple[
        list[Chunk],
        float,
    ]:
        selected: list[
            Chunk
        ] = []
        used = 0.0

        while True:
            qc = [
                c
                for c in
                self.all_chunks
                if self.compute_ready(
                    c
                )
            ]
            qc.sort(
                key=lambda c: (
                    -self.compute_priority(
                        c
                    ),
                    self.comp_ms[c],
                    c,
                )
            )

            fitting = [
                c
                for c in qc
                if (
                    used
                    + self.comp_ms[c]
                    <= self.delta_ms
                    + 1e-9
                )
            ]

            if not fitting:
                break

            c = fitting[0]
            self.done[c] = "compute"
            selected.append(c)
            used += self.comp_ms[c]

            # Qc is deliberately reconstructed and re-sorted on the next
            # iteration because this compute may unlock more chunks.

        return selected, used

    def _stream_phase(
        self,
    ) -> tuple[
        list[Chunk],
        float,
    ]:
        qs = [
            c
            for c in
            self.all_chunks
            if c not in self.done
        ]
        qs.sort(
            key=lambda c: (
                -self.stream_priority(
                    c
                ),
                self.stream_ms[c],
                c,
            )
        )

        selected: list[
            Chunk
        ] = []
        used = 0.0

        for c in qs:
            cost = self.stream_ms[
                c
            ]
            if (
                used + cost
                <= self.delta_ms
                + 1e-9
            ):
                selected.append(c)
                used += cost

        for c in selected:
            self.done[c] = "stream"

        return selected, used

    def run(
        self,
    ) -> dict[str, Any]:
        self.reset()

        stages: list[
            dict[str, Any]
        ] = []
        makespan_ms = 0.0

        while (
            len(self.done)
            < len(
                self.all_chunks
            )
        ):
            stage_id = (
                len(stages) + 1
            )

            compute, compute_ms = (
                self._compute_phase()
            )

            # Paper resets the budget before streaming.
            stream, stream_ms = (
                self._stream_phase()
            )

            if (
                not compute
                and not stream
            ):
                remaining = sorted(
                    self.all_chunks
                    - set(
                        self.done
                    )
                )
                smallest_compute = min(
                    (
                        self.comp_ms[c]
                        for c in
                        remaining
                        if self.compute_ready(
                            c
                        )
                    ),
                    default=float(
                        "inf"
                    ),
                )
                smallest_stream = min(
                    self.stream_ms[c]
                    for c in remaining
                )

                raise RuntimeError(
                    "SparKV greedy scheduler made no progress. "
                    "Delta t is smaller than every feasible operation: "
                    f"delta_ms={self.delta_ms}, "
                    f"min_compute={smallest_compute}, "
                    f"min_stream={smallest_stream}. "
                    "Increase --delta-ms; the paper does not define an "
                    "oversized-operation exception."
                )

            duration = max(
                compute_ms,
                stream_ms,
            )
            makespan_ms += duration

            stages.append(
                {
                    "stage":
                        stage_id,
                    "duration_ms":
                        duration,
                    "compute_ms":
                        compute_ms,
                    "stream_ms":
                        stream_ms,
                    "compute": [
                        c.__dict__
                        for c in compute
                    ],
                    "stream": [
                        c.__dict__
                        for c in stream
                    ],
                }
            )

        compute_count = sum(
            route == "compute"
            for route in
            self.done.values()
        )

        return {
            "scheduler":
                "sparkv-potential-aware-greedy",
            "indexing":
                "zero-based-python",
            "delta_ms":
                self.delta_ms,
            "makespan_ms":
                makespan_ms,
            "chunks":
                len(
                    self.all_chunks
                ),
            "compute_chunks":
                compute_count,
            "stream_chunks":
                (
                    len(
                        self.all_chunks
                    )
                    - compute_count
                ),
            "stages":
                stages,
            "assignments": {
                c.key: route
                for c, route in
                sorted(
                    self.done.items()
                )
            },
            "unit_costs": {
                c.key: {
                    "comp_ms":
                        float(
                            self.comp_ms[
                                c
                            ]
                        ),
                    "stream_ms":
                        float(
                            self.stream_ms[
                                c
                            ]
                        ),
                }
                for c in sorted(
                    self.all_chunks
                )
            },
        }


def load_costs_from_profile(
    *,
    profile_path: Path,
    bandwidth_mbps: float,
) -> tuple[
    int,
    int,
    int,
    dict[Chunk, float],
    dict[Chunk, float],
]:
    if bandwidth_mbps <= 0:
        raise ValueError(
            "bandwidth_mbps must be positive"
        )

    profile = json.loads(
        profile_path.read_text(
            encoding="utf-8"
        )
    )

    geometry = profile[
        "geometry"
    ]
    T = int(
        geometry[
            "token_chunks"
        ]
    )
    L = int(
        geometry["layers"]
    )
    H = int(
        geometry["heads"]
    )

    units = profile[
        "units"
    ]

    comp: dict[
        Chunk,
        float,
    ] = {}
    stream: dict[
        Chunk,
        float,
    ] = {}

    for t in range(T):
        for layer in range(L):
            for head in range(H):
                c = Chunk(
                    t,
                    layer,
                    head,
                )
                item = units[
                    c.key
                ]

                comp[c] = float(
                    item[
                        "predicted_comp_ms"
                    ]
                )

                wire_bytes = int(
                    item[
                        "wire_bytes"
                    ]
                )
                processing_ms = float(
                    item[
                        "processing_ms"
                    ]
                )

                wire_ms = (
                    wire_bytes
                    * 8.0
                    / (
                        bandwidth_mbps
                        * 1e6
                    )
                    * 1000.0
                )

                stream[c] = (
                    wire_ms
                    + processing_ms
                )

    return (
        T,
        L,
        H,
        comp,
        stream,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        required=True,
    )
    parser.add_argument(
        "--bandwidth-mbps",
        type=float,
        default=640.0,
    )
    parser.add_argument(
        "--delta-ms",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    args = parser.parse_args()

    (
        T,
        L,
        H,
        comp,
        stream,
    ) = load_costs_from_profile(
        profile_path=Path(
            args.profile
        ),
        bandwidth_mbps=
            args.bandwidth_mbps,
    )

    scheduler = SparKVScheduler(
        token_chunks=T,
        layers=L,
        heads=H,
        comp_ms=comp,
        stream_ms=stream,
        delta_ms=args.delta_ms,
    )
    result = scheduler.run()

    result[
        "profile_path"
    ] = str(
        args.profile
    )
    result[
        "bandwidth_mbps"
    ] = float(
        args.bandwidth_mbps
    )

    output = Path(
        args.output
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "saved":
                    str(output),
                "makespan_ms":
                    result[
                        "makespan_ms"
                    ],
                "compute_chunks":
                    result[
                        "compute_chunks"
                    ],
                "stream_chunks":
                    result[
                        "stream_chunks"
                    ],
                "stages":
                    len(
                        result[
                            "stages"
                        ]
                    ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
