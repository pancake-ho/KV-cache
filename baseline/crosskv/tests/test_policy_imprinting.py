import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from xmodel_kv.cli.bfcl_memory_policy_imprinting import _memory_document_cases
from xmodel_kv.cli.bfcl_policy_imprinting import (
    _background_history,
    _handoff_history,
    _handoff_prompts,
)

from xmodel_kv.policy_imprinting import (
    action_logits,
    build_chat_segments,
    compare_action_distributions,
    history_message_spans,
    layerwise_history_distance,
    parse_tool_call,
    patch_history_cache,
    prefill_legacy_cache,
    start_readout,
    stitch_history_cache,
    validate_shared_handoff,
)


def _tiny_qwen() -> Qwen3ForCausalLM:
    torch.manual_seed(12)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=1_000_000,
        attention_dropout=0.0,
        use_cache=True,
    )
    return Qwen3ForCausalLM(config).eval()


def test_identity_stitch_reproduces_native_action_logits():
    model = _tiny_qwen()
    prefix = (4, 5, 6, 7)
    history = (8, 9, 10, 11, 12)
    readout = (13, 14)
    context_cache = prefill_legacy_cache(model, prefix + history)
    prefix_cache = prefill_legacy_cache(model, prefix)
    stitched = stitch_history_cache(
        model,
        source_context_cache=context_cache,
        target_prefix_cache=prefix_cache,
        source_prefix_length=len(prefix),
        target_prefix_length=len(prefix),
        transfer_history_length=len(history),
    )

    native_factory = lambda: start_readout(
        model,
        legacy_cache=context_cache,
        cached_tokens=len(prefix) + len(history),
        input_ids=readout,
    )
    stitched_factory = lambda: start_readout(
        model,
        legacy_cache=stitched,
        cached_tokens=len(prefix) + len(history),
        input_ids=readout,
    )
    action = (15, 16, 17)
    native_logits = action_logits(model, native_factory, action)
    stitched_logits = action_logits(model, stitched_factory, action)
    # CPU attention kernels may reorder float32 reductions across runners.
    torch.testing.assert_close(stitched_logits, native_logits, atol=2e-7, rtol=1e-6)
    comparison = compare_action_distributions(native_logits, stitched_logits, action)
    assert comparison.mean_kl == 0
    assert comparison.top1_agreement == 1


def test_layer_zero_history_is_prefix_independent_after_rope_removal():
    model = _tiny_qwen()
    source_prefix = (4, 5, 6, 7)
    target_prefix = (21, 22, 23, 24)
    history = (8, 9, 10, 11, 12)
    source_cache = prefill_legacy_cache(model, source_prefix + history)
    target_cache = prefill_legacy_cache(model, target_prefix + history)
    distances = layerwise_history_distance(
        model,
        source_context_cache=source_cache,
        target_context_cache=target_cache,
        source_prefix_length=len(source_prefix),
        target_prefix_length=len(target_prefix),
        history_length=len(history),
    )
    assert distances[0]["key_relative_l2"] == 0
    assert distances[0]["value_relative_l2"] == 0
    assert distances[1]["key_relative_l2"] > 0
    assert distances[1]["value_relative_l2"] > 0


def test_repositioned_layer_zero_history_matches_new_absolute_positions():
    model = _tiny_qwen()
    source_prefix = (4, 5, 6)
    target_prefix = (21, 22, 23, 24, 25)
    history = (8, 9, 10, 11)
    source_cache = prefill_legacy_cache(model, source_prefix + history)
    target_cache = prefill_legacy_cache(model, target_prefix + history)
    target_prefix_cache = prefill_legacy_cache(model, target_prefix)
    stitched = stitch_history_cache(
        model,
        source_context_cache=source_cache,
        target_prefix_cache=target_prefix_cache,
        source_prefix_length=len(source_prefix),
        target_prefix_length=len(target_prefix),
        transfer_history_length=len(history),
    )
    stitched_key, stitched_value = stitched[0]
    target_key, target_value = target_cache[0]
    torch.testing.assert_close(
        stitched_key[:, :, len(target_prefix) :],
        target_key[:, :, len(target_prefix) :],
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        stitched_value[:, :, len(target_prefix) :],
        target_value[:, :, len(target_prefix) :],
        atol=1e-7,
        rtol=1e-5,
    )


def test_component_and_token_patching_changes_only_selected_cache_entries():
    model = _tiny_qwen()
    source_prefix = (4, 5, 6, 7)
    target_prefix = (21, 22, 23, 24)
    history = (8, 9, 10, 11, 12)
    source_cache = prefill_legacy_cache(model, source_prefix + history)
    target_cache = prefill_legacy_cache(model, target_prefix + history)
    target_prefix_cache = prefill_legacy_cache(model, target_prefix)
    mask = (False, True, False, False, False)
    patched = patch_history_cache(
        model,
        source_context_cache=source_cache,
        target_context_cache=target_cache,
        target_prefix_cache=target_prefix_cache,
        source_prefix_length=len(source_prefix),
        target_prefix_length=len(target_prefix),
        history_length=len(history),
        source_layers=(1,),
        source_keys=False,
        source_values=True,
        source_token_mask=mask,
    )
    for layer, ((patched_key, patched_value), (target_key, target_value)) in enumerate(
        zip(patched, target_cache, strict=True)
    ):
        target_history_key = target_key[:, :, len(target_prefix) :]
        target_history_value = target_value[:, :, len(target_prefix) :]
        torch.testing.assert_close(
            patched_key[:, :, len(target_prefix) :], target_history_key
        )
        value_difference = (
            patched_value[:, :, len(target_prefix) :] - target_history_value
        ).abs()
        if layer == 1:
            assert value_difference[:, :, 1].max() > 0
            assert value_difference[:, :, (0, 2, 3, 4)].max() == 0
        else:
            assert value_difference.max() == 0
class _TemplateTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        add_generation_prompt,
        enable_thinking,
        tokenize,
    ):
        del tools, enable_thinking, tokenize
        ids = []
        for message in messages:
            ids.extend([1 if message["role"] == "system" else 2])
            ids.extend(ord(character) for character in message["content"])
            ids.append(3)
        if add_generation_prompt:
            ids.extend((4, 5))
        return ids


def test_chat_segments_require_shared_history_and_readout():
    tokenizer = _TemplateTokenizer()
    history = [{"role": "user", "content": "same history"}]
    source = build_chat_segments(
        tokenizer, system_prompt="source", tools=[], history=history
    )
    target = build_chat_segments(
        tokenizer, system_prompt="target", tools=[], history=history
    )
    validate_shared_handoff(source, target)
    assert source.prefix_ids != target.prefix_ids
    assert source.history_ids == target.history_ids
    assert source.readout_ids == target.readout_ids == (4, 5)
    spans = history_message_spans(
        tokenizer,
        system_prompt="source",
        tools=[],
        history=[
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ],
    )
    assert spans[0][0] == 0
    assert spans[0][1] == 5
    assert spans[1] == (5, 10)


class _FirstUserToolTokenizer:
    marker = "<|reserved_special_token_0|>"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        if text == self.marker:
            return [999]
        return [ord(character) for character in text]

    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        add_generation_prompt,
        enable_thinking,
        tokenize,
    ):
        del enable_thinking, tokenize
        if tools and not any(message["role"] == "user" for message in messages):
            raise RuntimeError("tools require a first user message")
        ids = []
        for index, message in enumerate(messages):
            ids.append(10 if message["role"] == "system" else 11)
            if index == 1 and tools:
                ids.extend((20, 21))
            content = message["content"]
            if content.startswith(self.marker):
                ids.append(999)
                content = content[len(self.marker) :]
            ids.extend(ord(character) for character in content)
            ids.append(12)
        if add_generation_prompt:
            ids.extend((13, 14))
        return ids


def test_chat_segments_fall_back_to_first_user_content_for_tool_templates():
    tokenizer = _FirstUserToolTokenizer()
    history = [
        {"role": "user", "content": "shared one"},
        {"role": "assistant", "content": "shared two"},
    ]
    tools = [{"type": "function", "function": {"name": "save"}}]
    source = build_chat_segments(
        tokenizer, system_prompt="source policy", tools=tools, history=history
    )
    target = build_chat_segments(
        tokenizer, system_prompt="target policy", tools=tools, history=history
    )
    validate_shared_handoff(source, target)
    assert source.prefix_ids != target.prefix_ids
    assert source.history_ids[0] == ord("s")
    assert source.readout_ids == (13, 14)
    spans = history_message_spans(
        tokenizer,
        system_prompt="source policy",
        tools=tools,
        history=history,
    )
    assert spans[0][0] == 0
    assert spans[-1][1] == len(source.history_ids)


def test_parse_tool_call_canonicalizes_json():
    call = parse_tool_call(
        '<tool_call>\n{"name":"route_blue","arguments":{"record_id":"R-104"}}\n</tool_call>'
    )
    assert call.valid_json
    assert call.name == "route_blue"
    assert call.arguments == {"record_id": "R-104"}
    assert not parse_tool_call("not a tool call").valid_json


def test_parse_tool_call_canonicalizes_llama_bare_json_parameters():
    call = parse_tool_call(
        '<|python_tag|>{"name":"route_blue","parameters":{"record_id":"R-104"}}'
        "<|eom_id|>"
    )
    assert call.valid_json
    assert call.name == "route_blue"
    assert call.arguments == {"record_id": "R-104"}


def test_bfcl_memory_case_mirrors_policy_while_sharing_tagged_history():
    document = {
        "id": "memory_prereq_test-notetaker-0",
        "topic": "notes",
    }
    tool = {
        "type": "function",
        "function": {
            "name": "archival_memory_add",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }
    cases, metadata = _memory_document_cases(
        document,
        domain="notetaker",
        records=["Monday note", "Friday note"],
        tool=tool,
        mode="tagged",
    )
    forward = cases["forward"]
    reverse = cases["reverse"]
    assert forward.history == reverse.history
    assert forward.source_tools == forward.target_tools
    assert forward.source_expected == reverse.target_expected
    assert forward.target_expected == reverse.source_expected
    assert forward.source_expected != forward.target_expected
    assert metadata["record_count"] == 2
    assert forward.history[-1]["content"].startswith("The memory batch is complete")


def test_bfcl_planner_executor_handoff_preserves_proposal_and_separates_roles():
    call = {"name": "lookup_ticket", "arguments": {"ticket_id": "T-7"}}
    history = _handoff_history([{"role": "user", "content": "Find T-7"}], call)
    planner, executor = _handoff_prompts()
    assert '"name":"lookup_ticket"' in history[-2]["content"]
    assert "already been handled" in history[-2]["content"]
    assert "initial-action planner" in planner
    assert "final-stage executor" in executor
    automatic = _handoff_history(
        [{"role": "user", "content": "Find T-7"}], call, include_trigger=False
    )
    assert automatic[-1]["role"] == "assistant"
    assert len(automatic) == 2


def test_bfcl_background_history_zero_is_empty(tmp_path):
    assert _background_history(tmp_path, 0) == []
