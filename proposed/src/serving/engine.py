"""Execute requests using a native backend and optional prefix reuse."""

from time import perf_counter

from .request import Request


def elapsed_ms(start):
    return (perf_counter() - start) * 1000


class ServingEngine:
    def __init__(self, runtime, *, cache=None, transfer=None):
        self.runtime = runtime
        self.cache = cache
        self.transfer = transfer

    def serve(self, request: Request, budget: int):
        if budget < 1 or not request.token_ids:
            raise ValueError("request and generation budget must be nonempty")
        started = perf_counter()
        with self.runtime.adapters.activate(request.model_id, request.adapter_id) as (model, adapter_sha256):
            self.runtime.sync()
            return self._serve(request, budget, model, adapter_sha256, started)

    def _serve(self, request, budget, model, adapter_sha256, started):
        cache = self.cache
        compressed = cache is not None and cache.codec is not None
        if len(request.token_ids) + budget > model.config.max_position_embeddings:
            raise ValueError("prompt and generation exceed model context")

        routing_ms = elapsed_ms(started)
        lookup = None
        if cache is not None:
            lookup = cache.lookup(request, adapter_sha256=adapter_sha256, transfer=self.transfer)
        entry = lookup.entry if lookup is not None else None
        prefix = 0
        past = None
        map_ms = 0.0
        cache_copy_ms = 0.0
        cache_decode_ms = 0.0
        path = "full_recompute" if cache is None else "miss_recompute"
        if entry is not None:
            prefix = len(entry.request.token_ids)
            if compressed:
                tick = perf_counter()
                layers = cache.restore(entry)
                self.runtime.sync()
                cache_decode_ms = elapsed_ms(tick)
            else:
                layers = entry.layers
            tick = perf_counter()
            if entry.request.model_id == request.model_id:
                past = self.runtime.copy_cache(layers, clone=not compressed)
                path = "native_kv_hit"
            else:
                past = self.transfer.map_cache(
                    layers, source=entry.request, target=request,
                    source_model=self.runtime.models[entry.request.model_id], target_model=model,
                )
                path = "cross_model_kv_hit"
            del layers
            self.runtime.layers(model, past, prefix)
            self.runtime.sync()
            if path == "cross_model_kv_hit":
                map_ms = elapsed_ms(tick)
            else:
                cache_copy_ms = elapsed_ms(tick)

        tick = perf_counter()
        suffix = request.token_ids[prefix:]
        result = self.runtime.forward(model, suffix, prefix, past)
        prefill = result.past_key_values
        generated = [int(result.logits[0, -1].argmax().item())]
        prefill_ms = elapsed_ms(tick)
        ttft_ms = elapsed_ms(started)
        del result, past

        snapshot = None
        snapshot_ms = 0.0
        if cache is not None and entry is None:
            tick = perf_counter()
            snapshot = cache.snapshot(
                request, self.runtime.layers(model, prefill, len(request.token_ids)), adapter_sha256,
            )
            self.runtime.sync()
            snapshot_ms = elapsed_ms(tick)

        tick = perf_counter()
        for step in range(budget - 1):
            result = self.runtime.forward(model, (generated[-1],), len(request.token_ids) + step, prefill)
            prefill = result.past_key_values
            generated.append(int(result.logits[0, -1].argmax().item()))
        decode_ms = elapsed_ms(tick)

        tick = perf_counter()
        if snapshot is not None:
            cache.admit(snapshot, lookup.cluster_id)
        admission_ms = elapsed_ms(tick) if snapshot is not None else 0.0
        text = self.runtime.tokenizers[request.model_id].decode(generated, skip_special_tokens=True)
        return {
            "request_id": request.request_id, "user_id": request.user_id, "model_id": request.model_id,
            "adapter_id": request.adapter_id, "adapter_sha256": adapter_sha256,
            "path": path, "reused_from": entry.request.request_id if entry else None,
            "prompt_tokens": len(request.token_ids), "reused_prefix_tokens": prefix,
            "forwarded_prefill_tokens": len(suffix), "generated_ids": generated, "generated_text": text,
            "cluster_id": lookup.cluster_id if lookup else None,
            "semantic_candidates": lookup.candidates if lookup else 0,
            "rejections": lookup.rejected if lookup else {},
            "semantic_encoding_ms": lookup.semantic_encoding_ms if lookup else 0.0,
            "cluster_assignment_ms": lookup.cluster_assignment_ms if lookup else 0.0,
            "candidate_lookup_ms": lookup.candidate_lookup_ms if lookup else 0.0,
            "model_routing_ms": routing_ms,
            "mapping_ms": map_ms, "same_model_copy_ms": cache_copy_ms,
            "prefill_ms": prefill_ms, "ttft_ms": ttft_ms, "decode_ms": decode_ms,
            "snapshot_ms": snapshot_ms, "admission_ms": admission_ms,
            "cache_encode_ms": snapshot_ms if compressed and snapshot is not None else 0.0,
            "cache_decode_ms": cache_decode_ms,
            "application_ms": elapsed_ms(started),
            "resident_kv_bytes": cache.bytes if cache else 0,
        }
