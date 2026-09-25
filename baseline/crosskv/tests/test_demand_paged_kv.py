import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from xmodel_kv.demand_paged_kv import (
    DemandPagedKVConfig,
    DemandPagedKVController,
    LayerPageStore,
    register_demand_paged_attention,
)


def _manual_attention(query, key, value):
    scores = torch.matmul(query.float(), key.float().transpose(2, 3))
    scores = scores / query.shape[-1] ** 0.5
    return torch.matmul(torch.softmax(scores, dim=-1), value.float())


def test_full_page_budget_matches_full_attention():
    generator = torch.Generator().manual_seed(17)
    key = torch.randn(1, 2, 11, 8, generator=generator)
    value = torch.randn(1, 2, 11, 8, generator=generator)
    query = torch.randn(1, 4, 1, 8, generator=generator)
    current_key = torch.randn(1, 2, 1, 8, generator=generator)
    current_value = torch.randn(1, 2, 1, 8, generator=generator)
    store = LayerPageStore(
        key,
        value,
        cold_start=2,
        cold_end=9,
        config=DemandPagedKVConfig(
            page_size=3, budget_ratio=1.0, pin_cpu_memory=False
        ),
        device=torch.device("cpu"),
    )

    actual, fetch = store.attend(
        query,
        current_key,
        current_value,
        step=0,
        layer=0,
        scaling=8**-0.5,
    )

    repeated_key = torch.cat((key, current_key), dim=2).repeat_interleave(2, dim=1)
    repeated_value = torch.cat((value, current_value), dim=2).repeat_interleave(
        2, dim=1
    )
    expected = _manual_attention(query, repeated_key, repeated_value)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
    assert fetch.selected_tokens == 7
    assert fetch.page_misses == 3


def test_sign_aware_directory_selects_page_with_largest_possible_logit():
    key = torch.zeros(1, 1, 8, 2)
    value = torch.arange(16, dtype=torch.float32).view(1, 1, 8, 2)
    key[:, :, 4:6, 0] = 9
    store = LayerPageStore(
        key,
        value,
        cold_start=0,
        cold_end=8,
        config=DemandPagedKVConfig(
            page_size=2,
            budget_ratio=0.25,
            pin_cpu_memory=False,
        ),
        device=torch.device("cpu"),
    )

    assert store.select_pages(torch.tensor([[[[1.0, 0.0]]]])) == [2]


def test_mandatory_pages_share_budget_with_dynamic_selection():
    key = torch.zeros(1, 1, 8, 2)
    value = torch.ones_like(key)
    key[:, :, 4:6, 0] = 9
    store = LayerPageStore(
        key,
        value,
        cold_start=0,
        cold_end=8,
        config=DemandPagedKVConfig(
            page_size=2,
            budget_ratio=0.5,
            pin_cpu_memory=False,
        ),
        device=torch.device("cpu"),
        mandatory_page_ids=[3],
    )

    selected = store.select_pages(torch.tensor([[[[1.0, 0.0]]]]))

    assert set(selected) == {2, 3}


def test_repeated_selection_hits_bounded_working_set():
    key = torch.zeros(1, 1, 4, 2)
    value = torch.ones_like(key)
    key[:, :, :2, 0] = 5
    store = LayerPageStore(
        key,
        value,
        cold_start=0,
        cold_end=4,
        config=DemandPagedKVConfig(
            page_size=2,
            budget_ratio=0.5,
            pin_cpu_memory=False,
        ),
        device=torch.device("cpu"),
    )
    query = torch.tensor([[[[1.0, 0.0]]]])
    current = torch.zeros(1, 1, 1, 2)

    _, first = store.attend(
        query, current, current, step=0, layer=0, scaling=2**-0.5
    )
    _, second = store.attend(
        query, current, current, step=1, layer=0, scaling=2**-0.5
    )

    assert first.page_misses == 1
    assert first.loaded_bytes > 0
    assert second.page_hits == 1
    assert second.loaded_bytes == 0
    assert len(store.cached_pages) == 1


def test_full_budget_decoder_path_matches_normal_prefill():
    register_demand_paged_attention()
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
    )
    config._attn_implementation = "demand_paged"
    model = LlamaForCausalLM(config).eval()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])

    with torch.inference_mode():
        expected = model(prompt, use_cache=False).logits[:, -1]
        prefix = model(prompt[:, :-1], use_cache=True)
        controller = DemandPagedKVController.from_cache(
            prefix.past_key_values,
            cold_start=1,
            cold_end=3,
            config=DemandPagedKVConfig(
                page_size=1, budget_ratio=1.0, pin_cpu_memory=False
            ),
            device="cpu",
        )
        controller.attach(model)
        actual = model(
            prompt[:, -1:],
            attention_mask=torch.ones((1, 5), dtype=torch.long),
            position_ids=torch.tensor([[4]]),
            cache_position=torch.tensor([4]),
            use_cache=False,
        ).logits[:, -1]
        controller.detach(model)

    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"page_size": 0},
        {"budget_ratio": 0},
        {"budget_ratio": 1.1},
        {"min_pages": 0},
        {"cache_capacity_ratio": 0.5},
    ),
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DemandPagedKVConfig(**kwargs)
