from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from baseline.sparkv.scheduler import Chunk


@dataclass(frozen=True)
class RuntimeControllerConfig:
    """
    The paper specifies the migration *policy* and a per-stage migration cap,
    but not the numerical window length, imbalance threshold, or cap.  These
    remain explicit reproducibility parameters.
    """
    window: int = 4
    imbalance_margin: float = 0.05
    max_migrations_per_stage: int = 4

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError(
                "window must be positive"
            )
        if self.imbalance_margin < 0:
            raise ValueError(
                "imbalance_margin must be non-negative"
            )
        if self.max_migrations_per_stage < 0:
            raise ValueError(
                "max_migrations_per_stage must be non-negative"
            )


class RuntimeController:
    """
    Section IV-D runtime adaptation.

    - Compute contention:
        first prefetch stream chunks from the next stage;
        then migrate the tail of current compute order to streaming.
    - Bandwidth degradation:
        move compute-ready current stream chunks to compute;
        then speculatively advance compute-ready stream chunks from next stage.
    """

    def __init__(
        self,
        *,
        layers: int,
        config: RuntimeControllerConfig,
    ) -> None:
        self.layers = int(
            layers
        )
        self.config = config

        self.compute_ratios = deque(
            maxlen=config.window
        )
        self.stream_ratios = deque(
            maxlen=config.window
        )

    @staticmethod
    def _mean(
        values: deque,
    ) -> float:
        if not values:
            return 1.0
        return float(
            sum(values)
            / len(values)
        )

    @property
    def compute_ratio(self) -> float:
        return self._mean(
            self.compute_ratios
        )

    @property
    def stream_ratio(self) -> float:
        return self._mean(
            self.stream_ratios
        )

    def observe_stage(
        self,
        *,
        predicted_compute_ms: float,
        actual_compute_ms: float,
        predicted_stream_ms: float,
        actual_stream_ms: float,
    ) -> None:
        if predicted_compute_ms > 0:
            self.compute_ratios.append(
                max(
                    0.0,
                    actual_compute_ms
                    / predicted_compute_ms,
                )
            )

        if predicted_stream_ms > 0:
            self.stream_ratios.append(
                max(
                    0.0,
                    actual_stream_ms
                    / predicted_stream_ms,
                )
            )

    def compute_bottleneck(
        self,
    ) -> bool:
        return (
            self.compute_ratio
            > self.stream_ratio
            * (
                1.0
                + self.config.imbalance_margin
            )
        )

    def stream_bottleneck(
        self,
    ) -> bool:
        return (
            self.stream_ratio
            > self.compute_ratio
            * (
                1.0
                + self.config.imbalance_margin
            )
        )

    def _token_ready(
        self,
        c: Chunk,
        done: dict[
            Chunk,
            str,
        ],
    ) -> bool:
        if (
            c.t == 0
            or c.layer
            == self.layers - 1
        ):
            return True
        return Chunk(
            c.t - 1,
            c.layer,
            c.head,
        ) in done

    def _layer_ready(
        self,
        c: Chunk,
        done: dict[
            Chunk,
            str,
        ],
    ) -> bool:
        if c.layer == 0:
            return True
        return (
            done.get(
                Chunk(
                    c.t,
                    c.layer - 1,
                    c.head,
                )
            )
            == "compute"
        )

    def compute_ready(
        self,
        c: Chunk,
        done: dict[
            Chunk,
            str,
        ],
    ) -> bool:
        return (
            c not in done
            and self._token_ready(
                c,
                done,
            )
            and self._layer_ready(
                c,
                done,
            )
        )

    @staticmethod
    def _to_chunk(
        item: dict[str, Any],
    ) -> Chunk:
        return Chunk(
            int(item["t"]),
            int(item["layer"]),
            int(item["head"]),
        )

    def adapt(
        self,
        *,
        current_stage: dict[str, Any],
        next_stage: dict[str, Any] | None,
        done: dict[
            Chunk,
            str,
        ],
        unit_costs: dict[
            str,
            dict[str, float],
        ] | None = None,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any] | None,
        dict[str, Any],
    ]:
        current = deepcopy(
            current_stage
        )
        following = (
            None
            if next_stage is None
            else deepcopy(
                next_stage
            )
        )

        migrations = 0
        events: list[
            dict[str, Any]
        ] = []

        cap = (
            self.config
            .max_migrations_per_stage
        )

        def route_cost(
            item: dict[str, Any],
            route: str,
        ) -> float:
            if unit_costs is None:
                return 1.0
            c = self._to_chunk(item)
            field = (
                "comp_ms"
                if route == "compute"
                else "stream_ms"
            )
            return float(
                unit_costs[
                    c.key
                ][field]
            )

        compute_work = sum(
            route_cost(
                item,
                "compute",
            )
            for item in current.get(
                "compute",
                [],
            )
        ) * self.compute_ratio

        stream_work = sum(
            route_cost(
                item,
                "stream",
            )
            for item in current.get(
                "stream",
                [],
            )
        ) * self.stream_ratio

        if cap == 0:
            return (
                current,
                following,
                {
                    "events": [],
                    "compute_ratio":
                        self.compute_ratio,
                    "stream_ratio":
                        self.stream_ratio,
                },
            )

        if self.compute_bottleneck():
            # 1) Speculatively prefetch from the next stage until the
            # predicted stream path is no longer under-filled relative to the
            # compute path, or until the per-stage migration cap is reached.
            if following is not None:
                movable = list(
                    following.get(
                        "stream",
                        [],
                    )
                )

                for item in movable:
                    if migrations >= cap:
                        break
                    if (
                        unit_costs is not None
                        and stream_work
                        >= compute_work
                    ):
                        break

                    following[
                        "stream"
                    ].remove(
                        item
                    )
                    current.setdefault(
                        "stream",
                        [],
                    ).append(
                        item
                    )
                    migrations += 1
                    stream_work += (
                        route_cost(
                            item,
                            "stream",
                        )
                        * self.stream_ratio
                    )
                    events.append(
                        {
                            "type":
                                "next-stage-prefetch",
                            "chunk":
                                item,
                        }
                    )

            # 2) If the link still has less work than the compute path, move
            # chunks from the tail of the computation order to streaming.
            compute_list = current.get(
                "compute",
                [],
            )

            while (
                compute_list
                and migrations < cap
                and (
                    unit_costs is None
                    or stream_work
                    < compute_work
                )
            ):
                item = (
                    compute_list.pop()
                )
                current.setdefault(
                    "stream",
                    [],
                ).append(
                    item
                )
                migrations += 1

                compute_work = max(
                    0.0,
                    compute_work
                    - route_cost(
                        item,
                        "compute",
                    )
                    * self.compute_ratio,
                )
                stream_work += (
                    route_cost(
                        item,
                        "stream",
                    )
                    * self.stream_ratio
                )

                events.append(
                    {
                        "type":
                            "compute-to-stream-tail",
                        "chunk":
                            item,
                    }
                )

        elif self.stream_bottleneck():
            # 1) Current-stage streamed chunks that are already compute-ready.
            stream_items = list(
                current.get(
                    "stream",
                    [],
                )
            )

            for item in stream_items:
                if migrations >= cap:
                    break
                if (
                    unit_costs is not None
                    and compute_work
                    >= stream_work
                ):
                    break

                c = self._to_chunk(
                    item
                )
                if not self.compute_ready(
                    c,
                    done,
                ):
                    continue

                current[
                    "stream"
                ].remove(
                    item
                )
                current.setdefault(
                    "compute",
                    [],
                ).append(
                    item
                )
                migrations += 1

                stream_work = max(
                    0.0,
                    stream_work
                    - route_cost(
                        item,
                        "stream",
                    )
                    * self.stream_ratio,
                )
                compute_work += (
                    route_cost(
                        item,
                        "compute",
                    )
                    * self.compute_ratio
                )

                events.append(
                    {
                        "type":
                            "stream-to-compute-current",
                        "chunk":
                            item,
                    }
                )

            # 2) Advance compute-ready streamed chunks from the next stage.
            if (
                following is not None
                and migrations < cap
            ):
                stream_items = list(
                    following.get(
                        "stream",
                        [],
                    )
                )

                for item in stream_items:
                    if migrations >= cap:
                        break
                    if (
                        unit_costs is not None
                        and compute_work
                        >= stream_work
                    ):
                        break

                    c = self._to_chunk(
                        item
                    )
                    if not self.compute_ready(
                        c,
                        done,
                    ):
                        continue

                    following[
                        "stream"
                    ].remove(
                        item
                    )
                    current.setdefault(
                        "compute",
                        [],
                    ).append(
                        item
                    )
                    migrations += 1

                    compute_work += (
                        route_cost(
                            item,
                            "compute",
                        )
                        * self.compute_ratio
                    )

                    events.append(
                        {
                            "type":
                                "stream-to-compute-next",
                            "chunk":
                                item,
                        }
                    )

        return (
            current,
            following,
            {
                "events":
                    events,
                "migrations":
                    migrations,
                "compute_ratio":
                    self.compute_ratio,
                "stream_ratio":
                    self.stream_ratio,
                "estimated_compute_work_ms":
                    compute_work,
                "estimated_stream_work_ms":
                    stream_work,
                "compute_bottleneck":
                    self.compute_bottleneck(),
                "stream_bottleneck":
                    self.stream_bottleneck(),
            },
        )
