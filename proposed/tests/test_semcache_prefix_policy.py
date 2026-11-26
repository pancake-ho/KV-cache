"""CPU fixtures for cache ownership and serving routes with the SemCache policy."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semcache.edgelora.r5_semantic_index import SemanticIndex

from proposed.src.cache.prefix import PrefixCache
from proposed.src.cache.semcache_policy import SemCachePrefixPolicy
from proposed.src.kv_transfer.adapter_pair import AdapterPairKVTransfer
from proposed.src.serving.engine import ServingEngine
from proposed.src.serving.request import Request
from proposed.validation import semcache_runner
from proposed.validation.e2e_workload import Event
from proposed.validation.serving_e2e import run_trace


class Encoder:
    metadata = {"fixture": True}

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class Tensor:
    def __init__(self, ids):
        self.ids = ids

    def numel(self):
        return len(self.ids)

    def element_size(self):
        return 1

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.ids)


def request(name, ids=(1, 2, 3), *, model="source", group="shared", adapter=None):
    return Request(name, "q", model, " ".join(map(str, ids)), ids, group, adapter)


class Runtime:
    def __init__(self):
        config = SimpleNamespace(max_position_embeddings=32)
        self.models = {key: SimpleNamespace(config=config) for key in ("source", "target")}
        self.tokenizers = {key: self for key in self.models}
        self.adapters = self

    @contextmanager
    def activate(self, model_id, adapter_id):
        yield self.models[model_id], f"{adapter_id}-sha" if adapter_id else None

    def sync(self):
        pass

    def forward(self, model, ids, offset, past=None):
        assert offset == len(past or ())
        return SimpleNamespace(past_key_values=(past or ()) + tuple(ids), logits=self)

    def __getitem__(self, key):
        return self

    def argmax(self):
        return self

    def item(self):
        return 42

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))

    def copy_cache(self, layers, *, clone=True):
        return layers[0][0].ids

    def layers(self, model, past, length):
        assert len(past) == length
        return ((Tensor(past), Tensor(past)),)


class Codec:
    def encoded_size(self, layers):
        return len(layers[0][0].ids)

    def encode(self, layers):
        return layers[0][0].ids

    def decode(self, ids):
        return ((Tensor(ids), Tensor(ids)),)


class PrefixPolicyTests(unittest.TestCase):
    def cache(self, capacity=6, codec=None):
        policy = SemCachePrefixPolicy()
        return PrefixCache(SemanticIndex(Encoder()), capacity, codec=codec, policy=policy)

    def test_count_only_executed_reuses_and_keep_identity_guard(self):
        cache = self.cache()
        engine = ServingEngine(Runtime(), cache=cache, collect_cache_metrics=True)
        engine.serve(request("source"), 1)
        self.assertIsNone(cache.lookup(request("other", (1, 2, 3, 4), group="private")).entry)
        probe = request("probe", (1, 2, 3, 4))
        self.assertIsNotNone(cache.lookup(probe).entry)
        self.assertEqual(cache.policy.resident["source"].hits, 0)
        row = engine.serve(probe, 2)
        self.assertEqual((row["path"], row["forwarded_prefill_tokens"]), ("native_kv_hit", 1))
        self.assertEqual(cache.policy.resident["source"].hits, 1)
        self.assertEqual(row["cache_policy_metrics"]["admission"], "hit")

    def test_rejected_candidate_leaves_owner_intact_and_skips_copy(self):
        cache = self.cache()
        engine = ServingEngine(Runtime(), cache=cache, collect_cache_metrics=True)
        engine.serve(request("source"), 1)
        for i in range(3):
            engine.serve(request(f"hit{i}", (1, 2, 3, 4 + i)), 1)
        with patch.object(Tensor, "clone", side_effect=AssertionError("should not copy")):
            row = engine.serve(request("rare", (7, 8, 9)), 1)
        self.assertEqual(row["cache_policy_metrics"]["admission"], "below_policy_threshold")
        self.assertEqual((list(cache.entries), cache.bytes, cache.evictions), (["source"], 6, 0))

    def test_successful_eviction_keeps_index_bytes_and_policy_in_sync(self):
        cache = self.cache()
        engine = ServingEngine(Runtime(), cache=cache, collect_cache_metrics=True)
        engine.serve(request("source"), 1)
        row = engine.serve(request("next", (7, 8, 9)), 1)
        self.assertEqual(row["cache_policy_metrics"]["evictions"], 1)
        self.assertEqual(row["cache_policy_metrics"]["evicted_bytes"], 6)
        self.assertEqual(cache.bytes, 6)
        self.assertEqual(set(cache.entries), set(cache.index.entries))
        self.assertEqual(set(cache.entries), set(cache.policy.resident))

    def test_short_and_oversize_are_reported(self):
        engine = ServingEngine(Runtime(), cache=self.cache(), collect_cache_metrics=True)
        for ids, reason in [((1, 2), "short_prefix"), ((1, 2, 3, 4), "oversize")]:
            row = engine.serve(request(reason, ids), 1)
            self.assertEqual(row["cache_policy_metrics"]["admission"], reason)

    def test_compressed_directed_adapter_pair_through_e2e_entry_point(self):
        runtime = Runtime()
        endpoints = [
            {"model_id": model, "adapter_id": adapter, "adapter_sha256": f"{adapter}-sha"}
            for model, adapter in (("source", "adapter-a"), ("target", "adapter-b"))
        ]
        mapper = SimpleNamespace(pair=("source", "target"), map_prefix=lambda layers, ids, **_: ids)
        transfer = AdapterPairKVTransfer(mapper, *endpoints)
        source = request("source", adapter="adapter-a")
        target = request("qa", (1, 2, 3, 4), model="target", adapter="adapter-b")
        trace = [Event(source, "producer", {}, None),
                 Event(target, "cross", {"source_prefix_tokens": 3}, "source")]
        for name in ("lru", SemCachePrefixPolicy.name):
            case = {"cache_mib": 1, "cache_policy": name}
            rows = run_trace(runtime, "lora_fitted", transfer, lambda: SemanticIndex(Encoder()),
                             trace, case, 2, Codec())
            self.assertEqual(rows[1]["path"], "cross_model_kv_hit")
            self.assertEqual(rows[1]["reused_prefix_tokens"], 3)
            self.assertEqual(rows[1]["cache_policy_metrics"]["policy"], name)
        default = run_trace(runtime, "full", None, None, trace, {}, 2, None)
        self.assertEqual(default[1]["path"], "full_recompute")
        self.assertNotIn("cache_policy_metrics", default[1])

    def test_d5_dispatch_preserves_checkout_and_upstream_arguments(self):
        root = Path.cwd()
        with patch.object(semcache_runner, "d5_git_state") as state, \
                patch.object(semcache_runner.subprocess, "call", return_value=7) as child:
            result = semcache_runner.main(["--semcache-root", str(root), "l6b", "--help"])
        self.assertEqual(result, 7)
        state.assert_called_once_with(root)
        args, kwargs = child.call_args
        self.assertEqual(args[0][1:], [str(root / "scripts/165_r5_d5_l6b.py"), "--help"])
        self.assertEqual(kwargs["cwd"], root)
