import torch
from transformers import LlamaConfig, LlamaForCausalLM

from xmodel_kv.demand_paged_kv import (
    DemandPagedKVConfig,
    DemandPagedKVController,
    register_demand_paged_attention,
)
from xmodel_kv.rag_chunk_kv import (
    compose_chunk_bundle,
    generate_from_chunk_controller,
    precompute_chunk_bundle,
)


def make_model() -> LlamaForCausalLM:
    register_demand_paged_attention()
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    config._attn_implementation = "demand_paged"
    return LlamaForCausalLM(config).eval()


def test_sequential_chunk_cache_matches_normal_causal_generation() -> None:
    torch.manual_seed(11)
    model = make_model()
    system_ids = [1, 2]
    chunks = [("d1", [3, 4]), ("d2", [5, 6, 7])]
    suffix_ids = [8, 9]
    full_ids = system_ids + chunks[0][1] + chunks[1][1] + suffix_ids

    with torch.inference_mode():
        expected = model.generate(
            torch.tensor([full_ids]),
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )[0, len(full_ids) :].tolist()
        bundle = precompute_chunk_bundle(
            model,
            system_ids=system_ids,
            chunks=chunks,
            mode="sequential",
        )
        composed = compose_chunk_bundle(model, bundle)
        controller = DemandPagedKVController.from_cache(
            composed.layers,
            cold_start=composed.cold_start,
            cold_end=composed.cold_end,
            config=DemandPagedKVConfig(
                page_size=2,
                budget_ratio=1.0,
                pin_cpu_memory=False,
            ),
            device="cpu",
            cold_page_ranges=composed.page_ranges(2),
        )
        actual = generate_from_chunk_controller(
            model,
            controller,
            suffix_ids=suffix_ids,
            virtual_prefix_tokens=composed.cold_end,
            max_new_tokens=3,
            eos_token_id=None,
        )

    assert actual == expected
    assert composed.page_ranges(2) == ((0, 2), (2, 4), (4, 5))
    assert composed.relocated_tokens == 0


def test_independent_chunks_are_relocated_without_changing_values() -> None:
    torch.manual_seed(12)
    model = make_model()
    bundle = precompute_chunk_bundle(
        model,
        system_ids=[1, 2],
        chunks=[("d1", [3, 4]), ("d2", [5, 6, 7])],
        mode="independent",
    )

    composed = compose_chunk_bundle(model, bundle)

    assert composed.relocated_tokens == 3
    seed_pages = composed.select_seed_pages(
        document_token_ids=[3, 4, 5, 6, 7],
        query_token_ids=[7],
        page_size=2,
        budget_ratio=0.5,
        min_pages=1,
    )
    assert seed_pages == (0, 1)
    for layer_index, (_, source_value) in enumerate(bundle.chunks[1].layers):
        _, composed_value = composed.layers[layer_index]
        torch.testing.assert_close(composed_value[:, :, 4:7], source_value)

    reversed_cache = compose_chunk_bundle(model, bundle, chunk_order=[1, 0])
    assert [name for name, _, _ in reversed_cache.chunk_ranges] == ["d2", "d1"]
    for layer_index, (_, source_value) in enumerate(bundle.chunks[0].layers):
        _, reversed_value = reversed_cache.layers[layer_index]
        torch.testing.assert_close(reversed_value[:, :, 5:7], source_value)
