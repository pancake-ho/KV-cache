"""Tiny Qwen on CPU with the upstream R5 semantic index."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from semcache.edgelora.r5_semantic_index import SemanticIndex
from semcache.semantic.encoder import ControlledEncoder
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from proposed.src.cache.prefix import PrefixCache
from proposed.src.kv_transfer.base_model import BaseModelKVTransfer
from proposed.src.models.qwen3 import QwenRuntime
from proposed.src.serving.engine import ServingEngine
from proposed.src.serving.request import Request
from proposed.validation.semcache_smoke import replay, validate_reuse
from proposed.validation.workload import add_lora_requests, controlled_requests


class CharTokenizer:
    def encode(self, text, **_kwargs):
        return list(map(ord, text))

    def decode(self, ids, **_kwargs):
        return "".join(map(chr, ids))


class IdentityMapper:
    config = SimpleNamespace(source_model="source", target_model="target")

    def __init__(self):
        self.calls = 0

    def map_cache(self, source_cache, **_kwargs):
        self.calls += 1
        return DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in source_cache))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(2026)
        cfg = Qwen3Config(
            vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128,
        )
        cfg._attn_implementation = "sdpa"
        source = Qwen3ForCausalLM(cfg).eval()
        target = Qwen3ForCausalLM(cfg).eval()
        target.load_state_dict(source.state_dict())
        cls.tokenizers = {name: CharTokenizer() for name in ("source", "target")}
        cls.runtime = QwenRuntime({"source": source, "target": target}, cls.tokenizers, "cpu")
        cls.requests = controlled_requests("source", "target", cls.tokenizers, 24)
        cls.encoder = ControlledEncoder({req.text: [1, 0] for req in cls.requests})

    def test_source_is_not_recomputed_on_cross_model_hit(self):
        full = replay(self.runtime, self.requests, 3)
        mapper = IdentityMapper()
        source_steps = []
        target_steps = []

        def record(destination):
            def hook(_model, _args, kwargs):
                destination.append(kwargs["input_ids"].shape[-1])
            return hook

        source_hook = self.runtime.models["source"].register_forward_pre_hook(
            record(source_steps), with_kwargs=True,
        )
        target_hook = self.runtime.models["target"].register_forward_pre_hook(
            record(target_steps), with_kwargs=True,
        )
        try:
            rows = replay(self.runtime, self.requests, 3, encoder=self.encoder, mapper=mapper)
        finally:
            source_hook.remove()
            target_hook.remove()
        validate_reuse(rows, 24)
        self.assertEqual(mapper.calls, 1)
        self.assertEqual(source_steps, [24, 1, 1])
        self.assertEqual(target_steps, [24, 1, 1, 48, 1, 1])
        self.assertEqual([r["generated_ids"] for r in rows], [r["generated_ids"] for r in full])

    def test_same_model_decode_keeps_resident_prefix_unchanged(self):
        source, hit, _ = self.requests
        same_model = Request("same-model", hit.user_id, source.model_id, hit.text, hit.token_ids)
        cache = PrefixCache(SemanticIndex(self.encoder), 1 << 20)
        engine = ServingEngine(self.runtime, cache=cache)
        engine.serve(source, 3)
        stored = cache.entries[source.request_id].layers
        before = tuple(t.clone() for pair in stored for t in pair)
        row = engine.serve(same_model, 3)
        self.assertEqual(row["path"], "native_kv_hit")
        self.assertEqual(row["generated_ids"], ServingEngine(self.runtime).serve(same_model, 3)["generated_ids"])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, (t for p in stored for t in p))))

    def test_capacity_and_share_group(self):
        source, hit, miss = self.requests
        entry_bytes = 24 * 2 * 2 * 2 * 8 * 4
        cache = PrefixCache(SemanticIndex(self.encoder), entry_bytes)
        model = self.runtime.models[source.model_id]
        with torch.inference_mode():
            result = self.runtime.forward(model, source.token_ids, 0)
            layers = self.runtime.layers(model, result.past_key_values, len(source.token_ids))
        cache.admit(cache.snapshot(source, layers), 0)
        foreign = Request(hit.request_id, hit.user_id, hit.model_id, hit.text, hit.token_ids, "private")
        lookup = cache.lookup(foreign, transfer=BaseModelKVTransfer(IdentityMapper()))
        self.assertIsNone(lookup.entry)
        self.assertEqual(lookup.rejected, {"different_share_group": 1})
        other = Request("other", miss.user_id, source.model_id, source.text, source.token_ids)
        cache.admit(cache.snapshot(other, layers), 0)
        self.assertEqual(list(cache.entries), ["other"])
        self.assertEqual(list(cache.index.entries), ["other"])
        self.assertEqual(cache.bytes, entry_bytes)


    def test_real_peft_switching_and_adapter_aware_reuse(self):
        from peft import LoraConfig, get_peft_model

        cfg = self.runtime.models["target"].config
        source = Qwen3ForCausalLM(cfg).eval()
        target = Qwen3ForCausalLM(cfg).eval()
        for model in (source, target):
            model.load_state_dict(self.runtime.models["source"].state_dict())
        runtime = QwenRuntime({"source": source, "target": target}, self.tokenizers, "cpu")
        base_logits = runtime.forward(target, self.requests[1].token_ids, 0).logits
        with tempfile.TemporaryDirectory() as directory:
            for index, name in enumerate(("medical", "finance")):
                base = Qwen3ForCausalLM(cfg)
                base.name_or_path = "target"
                peft_model = get_peft_model(base, LoraConfig(
                    r=2, lora_alpha=2, target_modules=["v_proj"],
                    task_type="CAUSAL_LM", init_lora_weights=False,
                ))
                path = Path(directory) / str(index)
                peft_model.save_pretrained(path)
                runtime.adapters.load("target", name, path)
            with runtime.adapters.activate("target") as (base, fingerprint):
                self.assertIsNone(fingerprint)
                torch.testing.assert_close(runtime.forward(base, self.requests[1].token_ids, 0).logits, base_logits)
            logits = []
            for adapter in ("medical", "finance"):
                with runtime.adapters.activate("target", adapter) as (model, fingerprint):
                    self.assertEqual(len(fingerprint), 64)
                    logits.append(runtime.forward(model, self.requests[1].token_ids, 0).logits)
            self.assertFalse(torch.equal(*logits))
            requests = add_lora_requests(self.requests, ["medical", "finance"])
            encoder = ControlledEncoder({req.text: [1, 0] for req in requests})
            mapper = IdentityMapper()
            full = replay(runtime, requests, 3)
            rows = replay(runtime, requests, 3, encoder=encoder, mapper=mapper)
            validate_reuse(rows, 24)
            self.assertEqual(mapper.calls, 1)
            self.assertEqual(rows[4]["generated_ids"], full[4]["generated_ids"])
            self.assertEqual(rows[5]["generated_ids"], full[5]["generated_ids"])
            self.assertNotEqual(rows[4]["adapter_sha256"], rows[5]["adapter_sha256"])


if __name__ == "__main__":
    unittest.main()
