from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class Chunk:
    t: int
    layer: int
    head: int


class SparKVScheduler:
    """Potential-aware greedy scheduler from SparKV Section IV-B."""

    def __init__(
        self,
        token_chunks: int,
        layers: int,
        heads: int,
        comp_ms: dict[Chunk, float],
        stream_ms: dict[Chunk, float],
        delta_ms: float,
    ) -> None:
        self.T = token_chunks
        self.L = layers
        self.H = heads
        self.comp_ms = comp_ms
        self.stream_ms = stream_ms
        self.delta_ms = delta_ms
        self.all_chunks = {
            Chunk(t, layer, head)
            for t in range(self.T)
            for layer in range(self.L)
            for head in range(self.H)
        }
        self.done: dict[Chunk, str] = {}

    def token_ready(self, c: Chunk, done: dict[Chunk, str]) -> bool:
        if c.t == 0 or c.layer == self.L - 1:
            return True
        return Chunk(c.t - 1, c.layer, c.head) in done

    def layer_ready(self, c: Chunk, done: dict[Chunk, str]) -> bool:
        if c.layer == 0:
            return True
        parent = Chunk(c.t, c.layer - 1, c.head)
        return done.get(parent) == "compute"

    def compute_ready(self, c: Chunk, done: dict[Chunk, str] | None = None) -> bool:
        done = self.done if done is None else done
        return c not in done and self.token_ready(c, done) and self.layer_ready(c, done)

    def children(self, c: Chunk, operation: str) -> set[Chunk]:
        children: set[Chunk] = set()
        if c.t + 1 < self.T and c.layer < self.L - 1:
            children.add(Chunk(c.t + 1, c.layer, c.head))
        if operation == "compute" and c.layer + 1 < self.L:
            children.add(Chunk(c.t, c.layer + 1, c.head))
        return children

    def newly_unlocked(self, c: Chunk, operation: str) -> set[Chunk]:
        before = {x for x in self.children(c, operation) if self.compute_ready(x)}
        after_done = dict(self.done)
        after_done[c] = operation
        after = {
            x
            for x in self.children(c, operation)
            if self.compute_ready(x, after_done)
        }
        return after - before

    def compute_score(self, c: Chunk) -> float:
        potential = sum(1.0 / self.comp_ms[x] for x in self.newly_unlocked(c, "compute"))
        return 1.0 / self.comp_ms[c] + potential

    def stream_score(self, c: Chunk) -> float:
        potential = sum(1.0 / self.comp_ms[x] for x in self.newly_unlocked(c, "stream"))
        return 1.0 / self.stream_ms[c] + potential

    @staticmethod
    def select_with_budget(
        candidates: list[Chunk], costs: dict[Chunk, float], budget: float
    ) -> list[Chunk]:
        selected: list[Chunk] = []
        used = 0.0
        for c in candidates:
            if used + costs[c] <= budget + 1e-9:
                selected.append(c)
                used += costs[c]
        return selected

    def run(self) -> dict:
        stages = []
        makespan_ms = 0.0
        stage_id = 0

        while len(self.done) < len(self.all_chunks):
            stage_id += 1
            compute_selected: list[Chunk] = []
            compute_used = 0.0

            # Re-sort after every compute selection because dependencies change.
            while True:
                ready = [c for c in self.all_chunks if self.compute_ready(c)]
                ready.sort(key=lambda c: (-self.compute_score(c), c))
                fitting = [
                    c
                    for c in ready
                    if compute_used + self.comp_ms[c] <= self.delta_ms + 1e-9
                ]
                if not fitting:
                    break
                c = fitting[0]
                self.done[c] = "compute"
                compute_selected.append(c)
                compute_used += self.comp_ms[c]

            remaining = [c for c in self.all_chunks if c not in self.done]
            remaining.sort(key=lambda c: (-self.stream_score(c), c))
            stream_selected = self.select_with_budget(
                remaining, self.stream_ms, self.delta_ms
            )
            stream_used = sum(self.stream_ms[c] for c in stream_selected)
            for c in stream_selected:
                self.done[c] = "stream"

            # Prevent a deadlock if delta_ms is smaller than every chunk cost.
            if not compute_selected and not stream_selected:
                remaining = [c for c in self.all_chunks if c not in self.done]
                c = min(remaining, key=lambda x: self.stream_ms[x])
                self.done[c] = "stream"
                stream_selected = [c]
                stream_used = self.stream_ms[c]

            stage_ms = max(compute_used, stream_used)
            makespan_ms += stage_ms
            stages.append(
                {
                    "stage": stage_id,
                    "duration_ms": stage_ms,
                    "compute_ms": compute_used,
                    "stream_ms": stream_used,
                    "compute": [c.__dict__ for c in compute_selected],
                    "stream": [c.__dict__ for c in stream_selected],
                }
            )

        compute_count = sum(path == "compute" for path in self.done.values())
        return {
            "makespan_ms": makespan_ms,
            "stages": stages,
            "chunks": len(self.all_chunks),
            "compute_chunks": compute_count,
            "stream_chunks": len(self.all_chunks) - compute_count,
        }


def load_costs(
    profile_path: str,
    cache_meta_path: str,
    bandwidth_mbps: float,
    processing_ms: float,
):
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    meta = json.loads(Path(cache_meta_path).read_text(encoding="utf-8"))
    layer_costs = profile["token_layer_ms_median"]
    T = int(meta["num_chunks"])
    L = int(meta["layers"])
    H = int(meta["kv_heads"])
    if len(layer_costs) != T or len(layer_costs[0]) != L:
        raise ValueError("profile and cache metadata shapes do not match")

    comp_ms: dict[Chunk, float] = {}
    stream_ms: dict[Chunk, float] = {}
    for t in range(T):
        for layer in range(L):
            for head in range(H):
                c = Chunk(t, layer, head)
                # HF executes dense/shared layer operators jointly. Until a per-head
                # Sparge profile is supplied, apportion measured layer time equally.
                comp_ms[c] = max(1e-6, float(layer_costs[t][layer]) / H)
                wire_bytes = int(meta["chunks"][t]["lh_wire_bytes"][f"{layer}:{head}"])
                stream_ms[c] = max(
                    1e-6,
                    wire_bytes * 8 / (bandwidth_mbps * 1e6) * 1000
                    + processing_ms,
                )
    return T, L, H, comp_ms, stream_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--cache-meta", required=True)
    parser.add_argument("--bandwidth-mbps", type=float, default=640.0)
    parser.add_argument("--processing-ms", type=float, default=0.02)
    parser.add_argument("--delta-ms", type=float, default=5.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    T, L, H, comp, stream = load_costs(
        args.profile,
        args.cache_meta,
        args.bandwidth_mbps,
        args.processing_ms,
    )
    scheduler = SparKVScheduler(T, L, H, comp, stream, args.delta_ms)
    result = scheduler.run()
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ["makespan_ms", "chunks", "compute_chunks", "stream_chunks"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
