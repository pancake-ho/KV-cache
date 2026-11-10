"""Controlled hit, miss and LoRA requests used only by validation."""

from dataclasses import replace

from proposed.src.serving.request import Request


def controlled_requests(source_id, target_id, tokenizers, prefix_tokens, suffix_tokens=24):
    """Use a prefix of the actual tokenized target text to avoid BPE boundary drift."""
    if not 4 <= prefix_tokens <= 4096:
        raise ValueError("prefix_tokens must be in [4, 4096]")
    if not 1 <= suffix_tokens <= 4096:
        raise ValueError("suffix_tokens must be in [1, 4096]")
    total = prefix_tokens + suffix_tokens
    source = tokenizers[source_id]
    target = tokenizers[target_id]
    passage = (
        "The community science museum records local weather every morning. "
        "Volunteers compare the readings and publish a short summary each week. "
    )
    text = passage * (total // 12 + 2) + "\nQuestion: What does the museum record?"

    def tokenize(prompt):
        ids = tuple(target.encode(prompt, add_special_tokens=False))
        if tuple(source.encode(prompt, add_special_tokens=False)) != ids:
            raise ValueError("source and target tokenization must agree on the controlled prompt")
        return ids

    ids = tokenize(text)
    if len(ids) < total:
        raise ValueError("controlled prompt is too short for the requested prefix")
    prefix = ids[:prefix_tokens]
    hit = ids[:total]
    miss_text = "Correction: " + text
    miss_ids = tokenize(miss_text)[:len(hit)]

    def as_text(values):
        return target.decode(list(values), clean_up_tokenization_spaces=False)

    if miss_ids[:prefix_tokens] == prefix:
        raise ValueError("negative control unexpectedly has the source prefix")
    if len(miss_ids) != len(hit):
        raise ValueError("the two target requests must have equal prompt lengths")
    return (
        Request("source", "user-a", source_id, as_text(prefix), prefix),
        Request("hit", "user-b", target_id, as_text(hit), hit),
        Request("miss", "user-c", target_id, as_text(miss_ids), miss_ids),
    )


def add_lora_requests(requests, adapter_ids):
    if len(adapter_ids) != 2 or adapter_ids[0] == adapter_ids[1]:
        raise ValueError("the LoRA smoke requires two distinct adapter IDs")
    source, hit, _ = requests
    return (*requests,
        replace(source, request_id="lora-source", user_id="user-d", model_id=hit.model_id,
                adapter_id=adapter_ids[0]),
        replace(hit, request_id="lora-same", user_id="user-e", adapter_id=adapter_ids[0]),
        replace(hit, request_id="lora-other", user_id="user-f", adapter_id=adapter_ids[1]),
    )


def multi_user_requests(
    primary, tokenizers, prefix_tokens, suffix_tokens, *, extra_models, routes, adapters,
    cross_model_probes=(),
):
    """Build independent groups for model and adapter reuse controls."""
    requests = []
    expected = {}

    def pair(tag, source_id, target_id, *, source_adapter, target_adapter):
        source, hit, _ = controlled_requests(source_id, target_id, tokenizers, prefix_tokens, suffix_tokens)
        source = replace(source, request_id=f"{tag}-source", user_id=f"{tag}-user-a",
                         share_group=tag, adapter_id=source_adapter)
        hit = replace(hit, request_id=f"{tag}-hit", user_id=f"{tag}-user-b",
                      share_group=tag, adapter_id=target_adapter)
        requests.extend((source, hit))
        expected[source.request_id] = ("miss_recompute", None, None)
        if source_id == target_id:
            expected[hit.request_id] = ("native_kv_hit", source.request_id, None)
        else:
            # The physical adapter check precedes the model-pair check.
            expected[hit.request_id] = ("miss_recompute", None, "adapter_mismatch")
        return hit

    for adapter_id, (model_id, _, _) in adapters.items():
        hit = pair(
            f"adapter-{adapter_id}", model_id, model_id,
            source_adapter=adapter_id, target_adapter=adapter_id,
        )
        other = next((name for name, (base_id, _, _) in adapters.items()
                      if name != adapter_id and base_id == model_id), None)
        if other is not None:
            mismatch = replace(hit, request_id=f"{hit.share_group}-mismatch",
                               user_id=f"{hit.share_group}-user-c", adapter_id=other)
            requests.append(mismatch)
            expected[mismatch.request_id] = ("miss_recompute", None, "adapter_mismatch")
    for index, (source_adapter, target_adapter) in enumerate(cross_model_probes):
        if source_adapter not in adapters or target_adapter not in adapters:
            raise ValueError("cross-model probes require two registered adapter IDs")
        source_model = adapters[source_adapter][0]
        target_model = adapters[target_adapter][0]
        if source_model == target_model:
            raise ValueError("cross-model probe adapters must belong to different base models")
        pair(
            f"cross-model-probe-{index}", source_model, target_model,
            source_adapter=source_adapter, target_adapter=target_adapter,
        )

    return tuple(requests), expected


def multi_user_request_cases(requests):
    """Label each request by its relation to its group's cache source."""
    sources = {}
    cases = {}
    for request in requests:
        source = sources.setdefault(request.share_group, request)
        if request is source:
            case = "cold_source"
        elif request.model_id != source.model_id:
            case = "different_model"
        elif request.adapter_id != source.adapter_id:
            case = "different_adapter"
        else:
            case = "same_model_adapter"
        cases[request.request_id] = case
    return cases
