"""Execute requests using a native backend and optional prefix reuse."""

from time import perf_counter

from .request import Request


def elapsed_ms(start):
    return (perf_counter() - start) * 1000


class ServingEngine:
    def __init__(self, runtime, *, cache=None, transfer=None, collect_cache_metrics=False,
                 collect_mapping_metrics=False, collect_prefill_metrics=False):
        self.runtime = runtime
        self.cache = cache
        self.transfer = transfer
        self.collect_cache_metrics = collect_cache_metrics
        self.collect_mapping_metrics = collect_mapping_metrics
        self.collect_prefill_metrics = collect_prefill_metrics

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
        cache_metrics = {} if self.collect_cache_metrics else None
        if cache is not None:
            lookup = cache.lookup(request, adapter_sha256=adapter_sha256, transfer=self.transfer)
        entry = lookup.entry if lookup is not None else None
        prefix = 0
        past = None
        map_ms = 0.0
        mapping_stages = {}
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
                stage_events = {} if self.collect_mapping_metrics else None
                past = self.transfer.map_cache(
                    layers, source=entry.request, target=request,
                    source_model=self.runtime.models[entry.request.model_id], target_model=model,
                    **({"stage_events": stage_events} if stage_events is not None else {}),
                )
                path = "cross_model_kv_hit"
            del layers
            self.runtime.layers(model, past, prefix)
            self.runtime.sync()
            if path == "cross_model_kv_hit":
                map_ms = elapsed_ms(tick)
                if stage_events:
                    mapping_stages = {name: start.elapsed_time(end)
                                      for name, (start, end) in stage_events.items()}
            else:
                cache_copy_ms = elapsed_ms(tick)

        if self.collect_prefill_metrics:
            import torch

            forward_start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            forward_start.record()

        tick = perf_counter()
        suffix = request.token_ids[prefix:]
        result = self.runtime.forward(model, suffix, prefix, past)
        if self.collect_prefill_metrics:
            forward_end.record()
            first_token_started = perf_counter()
        prefill = result.past_key_values
        generated = [int(result.logits[0, -1].argmax().item())]
        token_ready = perf_counter()
        prefill_ms = (token_ready - tick) * 1000
        ttft_ms = (token_ready - started) * 1000
        prefill_stages = {}
        if self.collect_prefill_metrics:
            prefill_stages = {
                "forward_host_ms": (first_token_started - tick) * 1000,
                "forward_cuda_stream_ms": forward_start.elapsed_time(forward_end),
                "first_token_host_ms": (token_ready - first_token_started) * 1000,
            }
        del result, past

        snapshot = None
        snapshot_ms = 0.0
        if cache is not None and entry is None:
            tick = perf_counter()
            snapshot = cache.snapshot(
                request, self.runtime.layers(model, prefill, len(request.token_ids)), adapter_sha256,
                metrics=cache_metrics,
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
            cache.admit(snapshot, lookup.cluster_id, metrics=cache_metrics)
        admission_ms = elapsed_ms(tick) if snapshot is not None else 0.0
        text = self.runtime.tokenizers[request.model_id].decode(generated, skip_special_tokens=True)
        if entry is not None and cache.policy is not None:
            tick = perf_counter()
            cache.policy.reuse(entry)
            if cache_metrics is not None:
                cache_metrics["reuse_update_ms"] = elapsed_ms(tick)
        if cache_metrics is not None:
            cache_metrics["policy"] = ("disabled" if cache is None else
                                       "lru" if cache.policy is None else cache.policy.name)
            cache_metrics.setdefault("admission", "disabled" if cache is None else "hit")
            cache_metrics["observation_ms"] = lookup.policy_observation_ms if lookup else 0.0
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
            **({
                "candidate_discover_ms": lookup.candidate_discover_ms,
                "candidate_validate_ms": lookup.candidate_validate_ms,
            } if lookup and self.collect_prefill_metrics else {}),
            "model_routing_ms": routing_ms,
            "mapping_ms": map_ms, "same_model_copy_ms": cache_copy_ms,
            **({"mapping_stages_cuda_ms": mapping_stages} if mapping_stages else {}),
            "prefill_ms": prefill_ms, "ttft_ms": ttft_ms, "decode_ms": decode_ms,
            **({"prefill_stages_ms": prefill_stages} if prefill_stages else {}),
            "snapshot_ms": snapshot_ms, "admission_ms": admission_ms,
            "cache_encode_ms": snapshot_ms if compressed and snapshot is not None else 0.0,
            "cache_decode_ms": cache_decode_ms,
            **({"cache_policy_metrics": cache_metrics} if cache_metrics is not None else {}),
            "application_ms": elapsed_ms(started),
            "resident_kv_bytes": cache.bytes if cache else 0,
        }
