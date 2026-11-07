from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from xmodel_kv.artifact import LinearKVMapper, MapperConfig
from xmodel_kv.generation import generate_pairs
from xmodel_kv.multiple_choice import (
    hellaswag_context_and_choices,
    score_choices,
    score_token_continuation,
)


def _tiny_qwen() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=1_000_000,
        attention_dropout=0.0,
        use_cache=True,
    )
    return Qwen3ForCausalLM(config).eval()


def _identity_mapper(root: Path, model: Qwen3ForCausalLM) -> LinearKVMapper:
    config = model.config
    selected = [[layer] for layer in range(config.num_hidden_layers)]
    MapperConfig(
        source_model="tiny",
        target_model="tiny",
        source_layers=config.num_hidden_layers,
        target_layers=config.num_hidden_layers,
        source_kv_heads=config.num_key_value_heads,
        target_kv_heads=config.num_key_value_heads,
        source_head_dim=config.head_dim,
        target_head_dim=config.head_dim,
        selected_layers=selected,
        weight_dtype="float32",
    ).save(root)
    (root / "layers").mkdir()
    features = config.num_key_value_heads * config.head_dim
    identity = torch.eye(features).reshape(features, config.num_key_value_heads, config.head_dim)
    bias = torch.zeros(config.num_key_value_heads, config.head_dim)
    for layer in range(config.num_hidden_layers):
        save_file(
            {
                "k_weight": identity,
                "k_bias": bias,
                "v_weight": identity.clone(),
                "v_bias": bias.clone(),
            },
            root / "layers" / f"layer_{layer:03d}.safetensors",
        )
    return LinearKVMapper(root)


def test_identity_mapper_preserves_cache_and_continuation_logits(tmp_path):
    torch.manual_seed(3)
    model = _tiny_qwen()
    mapper = _identity_mapper(tmp_path, model)
    prefix = torch.randint(0, model.config.vocab_size, (1, 9))
    continuation = torch.randint(0, model.config.vocab_size, (1, 4))

    with torch.inference_mode():
        prefilling = model.model(input_ids=prefix, use_cache=True, return_dict=True)
        mapped = mapper.map_cache(
            prefilling.past_key_values,
            source_model=model,
            target_model=model,
        )
        original_layers = prefilling.past_key_values.to_legacy_cache()
        mapped_layers = mapped.to_legacy_cache()
        for (original_k, original_v), (mapped_k, mapped_v) in zip(original_layers, mapped_layers):
            torch.testing.assert_close(mapped_k, original_k, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(mapped_v, original_v, atol=2e-5, rtol=2e-5)

        original_logits = model(
            input_ids=continuation,
            past_key_values=prefilling.past_key_values,
            use_cache=False,
        ).logits
        mapped_logits = model(
            input_ids=continuation,
            past_key_values=mapped,
            use_cache=False,
        ).logits
    torch.testing.assert_close(mapped_logits, original_logits, atol=3e-5, rtol=3e-5)


def test_identity_mapper_preserves_left_padded_batched_cache(tmp_path):
    torch.manual_seed(5)
    model = _tiny_qwen()
    mapper = _identity_mapper(tmp_path, model)
    input_ids = torch.randint(2, model.config.vocab_size, (2, 9))
    attention_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1], [1] * 9])
    positions = attention_mask.cumsum(dim=1).sub(1).clamp_min(0)

    with torch.inference_mode():
        prefilling = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=positions,
            use_cache=True,
            return_dict=True,
        )
        mapped = mapper.map_cache(
            prefilling.past_key_values,
            source_model=model,
            target_model=model,
            positions=positions,
        )
    for (original_k, original_v), (mapped_k, mapped_v) in zip(
        prefilling.past_key_values.to_legacy_cache(), mapped.to_legacy_cache(), strict=True
    ):
        torch.testing.assert_close(mapped_k, original_k, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(mapped_v, original_v, atol=2e-5, rtol=2e-5)


class _CharacterTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + ord(character) % 95 for character in text]}

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids if token_id != self.pad_token_id)


def test_identity_mapper_preserves_batched_choice_scores(tmp_path):
    torch.manual_seed(4)
    model = _tiny_qwen()
    mapper = _identity_mapper(tmp_path, model)
    scores = score_choices(
        source_model=model,
        target_model=model,
        mapper=mapper,
        tokenizer=_CharacterTokenizer(),
        context="a context ",
        choices=["with one ending", "with a much longer ending", "third"],
    )
    torch.testing.assert_close(
        torch.tensor(scores.transfer),
        torch.tensor(scores.standalone),
        atol=5e-5,
        rtol=5e-5,
    )
    assert scores.character_counts == [15, 25, 5]
    assert scores.token_counts == [17, 27, 7]
    torch.testing.assert_close(
        torch.tensor(scores.standalone_normalized),
        torch.tensor(scores.standalone) / torch.tensor(scores.character_counts),
    )


def test_identity_mapper_preserves_left_padded_batched_generation(tmp_path):
    torch.manual_seed(6)
    model = _tiny_qwen()
    mapper = _identity_mapper(tmp_path, model)
    generations = generate_pairs(
        source_model=model,
        target_model=model,
        mapper=mapper,
        tokenizer=_CharacterTokenizer(),
        contexts=["a short context", "a substantially longer context"],
        max_new_tokens=5,
        stop=(),
    )
    assert [item.transfer for item in generations] == [item.standalone for item in generations]
    assert [item.transfer_tokens for item in generations] == [5, 5]


def test_hellaswag_uses_harness_target_delimiter():
    context, choices, label = hellaswag_context_and_choices(
        {
            "activity_label": "Opening a door",
            "ctx_a": "A person",
            "ctx_b": "opens a door",
            "endings": ["and enters."],
            "label": "0",
        }
    )
    assert context == "Opening a door: A person Opens a door"
    assert choices == ["and enters."]
    assert label == 0


def test_identity_mapper_preserves_fixed_token_continuation(tmp_path):
    torch.manual_seed(9)
    model = _tiny_qwen()
    mapper = _identity_mapper(tmp_path, model)
    scores = score_token_continuation(
        source_model=model,
        target_model=model,
        mapper=mapper,
        prefix_ids=[4, 5, 6, 7, 8],
        continuation_ids=[9, 10, 11, 12],
    )
    assert scores.tokens == 3
    assert abs(scores.standalone_log_likelihood - scores.transfer_log_likelihood) < 1e-4
