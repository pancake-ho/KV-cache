from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache
from transformers.generation.utils import (
    GenerationMixin,
)

from baseline.sparkv.experiment import (
    _generate_first_token,
)


class FakeCache:
    def __init__(
        self,
        length: int,
    ) -> None:
        self.length = int(length)

    def get_seq_length(
        self,
    ) -> int:
        return self.length


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2


class FakeRuntime:
    device = torch.device("cpu")


class RecordingModel:
    def __init__(
        self,
        next_token: int = 99,
    ) -> None:
        self.next_token = int(
            next_token
        )
        self.generate_kwargs = None

    def generate(
        self,
        **kwargs,
    ):
        self.generate_kwargs = kwargs
        input_ids = kwargs[
            "input_ids"
        ]
        next_token = torch.tensor(
            [[self.next_token]],
            dtype=torch.long,
            device=input_ids.device,
        )
        old_cache = kwargs[
            "past_key_values"
        ]
        new_cache = FakeCache(
            old_cache.get_seq_length()
            + 1
        )
        return SimpleNamespace(
            sequences=torch.cat(
                [
                    input_ids,
                    next_token,
                ],
                dim=1,
            ),
            past_key_values=
                new_cache,
        )


class MinimalGenerationMixin(
    GenerationMixin
):
    config = SimpleNamespace(
        is_encoder_decoder=False
    )


def _dynamic_cache(
    length: int,
) -> DynamicCache:
    key = torch.zeros(
        1,
        1,
        length,
        2,
    )
    value = torch.zeros_like(
        key
    )
    return (
        DynamicCache
        .from_legacy_cache(
            (
                (
                    key,
                    value,
                ),
            )
        )
    )


def test_hf_4513_seed_only_makes_empty_cache_position():
    mixin = (
        MinimalGenerationMixin()
    )
    out = (
        mixin
        ._get_initial_cache_position(
            torch.tensor(
                [[13]],
                dtype=torch.long,
            ),
            {
                "past_key_values":
                    _dynamic_cache(
                        3
                    )
            },
        )
    )
    assert (
        out[
            "cache_position"
        ].numel()
        == 0
    )


def test_hf_4513_full_logical_prompt_keeps_uncached_position():
    mixin = (
        MinimalGenerationMixin()
    )
    out = (
        mixin
        ._get_initial_cache_position(
            torch.tensor(
                [[10, 11, 12, 13]],
                dtype=torch.long,
            ),
            {
                "past_key_values":
                    _dynamic_cache(
                        3
                    )
            },
        )
    )
    assert (
        out[
            "cache_position"
        ].tolist()
        == [3]
    )


def test_generate_first_token_passes_full_logical_prompt():
    model = RecordingModel()
    cache = FakeCache(3)

    first_token, decode_cache = (
        _generate_first_token(
            model=model,
            tokenizer=FakeTokenizer(),
            cache=cache,
            prefill_ids=[
                10,
                11,
                12,
            ],
            seed_id=13,
            runtime=FakeRuntime(),
        )
    )

    assert first_token == 99
    assert (
        decode_cache
        .get_seq_length()
        == 4
    )
    assert (
        model.generate_kwargs[
            "input_ids"
        ].tolist()
        == [[10, 11, 12, 13]]
    )
    assert (
        model.generate_kwargs[
            "past_key_values"
        ]
        is cache
    )
    assert (
        model.generate_kwargs[
            "temperature"
        ]
        is None
    )


def test_generate_first_token_rejects_cache_prefix_mismatch():
    model = RecordingModel()

    with pytest.raises(
        RuntimeError,
        match=(
            "prefill/cache length "
            "mismatch"
        ),
    ):
        _generate_first_token(
            model=model,
            tokenizer=FakeTokenizer(),
            cache=FakeCache(2),
            prefill_ids=[
                10,
                11,
                12,
            ],
            seed_id=13,
            runtime=FakeRuntime(),
        )

    assert (
        model.generate_kwargs
        is None
    )
