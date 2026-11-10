"""Qwen3 native forward, cache layout and request-local cache copies."""

import torch
from transformers import DynamicCache

from .lora import LoRARegistry


class QwenRuntime:
    def __init__(self, models, tokenizers, device, adapters=None):
        self.models = models
        self.tokenizers = tokenizers
        self.device = torch.device(device)
        self.adapters = adapters if adapters is not None else LoRARegistry(models)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def forward(self, model, ids, offset, past=None):
        length = len(ids)
        return model(
            input_ids=torch.tensor([ids], dtype=torch.long, device=self.device),
            attention_mask=torch.ones((1, offset + length), dtype=torch.long, device=self.device),
            cache_position=torch.arange(offset, offset + length, device=self.device),
            past_key_values=past,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )

    def layers(self, model, past, length):
        layers = past.to_legacy_cache()
        config = model.config
        expected = (1, config.num_key_value_heads, length, config.head_dim)
        dtype = model.model.embed_tokens.weight.dtype
        if len(layers) != config.num_hidden_layers or any(
            t.shape != expected or t.device != self.device or t.dtype != dtype
            for pair in layers for t in pair
        ):
            raise RuntimeError(f"unexpected Qwen cache layout, expected {expected} on {self.device}")
        return layers

    @torch.inference_mode()
    def warmup(self):
        for model_id, tokenizer in self.tokenizers.items():
            ids = tokenizer.encode("Warmup.", add_special_tokens=False)
            with self.adapters.activate(model_id) as (model, _):
                self.forward(model, ids, 0)
        self.sync()

    @staticmethod
    @torch.inference_mode()
    def copy_cache(layers, *, clone=True):
        if clone:
            layers = tuple((k.clone(), v.clone()) for k, v in layers)
        return DynamicCache.from_legacy_cache(layers)
