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
