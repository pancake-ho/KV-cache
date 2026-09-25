import json

import numpy as np
import torch

from baseline.sparkv.overhead_model import (
    ComputationLatencyPredictor,
    PredictorTrainConfig,
    train_predictor,
)


def test_target_normalization_checkpoint_is_finite(tmp_path):
    rng = np.random.default_rng(
        7
    )
    records = []
    for index in range(
        128
    ):
        token = float(
            1
            + index % 8
        )
        active = float(
            8
            + index % 31
        )
        util = float(
            rng.integers(
                0,
                100,
            )
        )
        target = (
            0.3
            + 0.07 * token
            + 0.015 * active
            + 0.002 * util
        )
        records.append(
            {
                "token_block":
                    token,
                "active_blocks":
                    active,
                "gpu_util":
                    util,
                "attention_ms":
                    target,
            }
        )

    checkpoint = (
        tmp_path
        / "predictor.pt"
    )
    result = train_predictor(
        records=records,
        output=checkpoint,
        config=(
            PredictorTrainConfig(
                seed=7,
                samples=128,
                epochs=80,
                learning_rate=
                    1e-2,
                momentum=0.9,
            )
        ),
        dense_ms=4.0,
        final_projection_ms=1.0,
    )

    assert np.isfinite(
        result[
            "test_mae_ms"
        ]
    )

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert (
        "target_normalization"
        in payload
    )
    assert (
        payload[
            "target_normalization"
        ][
            "std"
        ]
        > 0
    )

    predictor = (
        ComputationLatencyPredictor(
            checkpoint
        )
    )
    value = predictor.attention_ms(
        token_block=4,
        active_blocks=20,
        gpu_util=50.0,
    )
    assert np.isfinite(
        value
    )
    assert value > 0
