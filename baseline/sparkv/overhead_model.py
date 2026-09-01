from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = (
    "token_block",
    "active_blocks",
    "gpu_util",
)


class SparseAttentionMLP(nn.Module):
    """
    Paper architecture: 3 inputs -> 48 -> 24 -> 1.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class PredictorTrainConfig:
    seed: int = 2026
    samples: int = 6000
    train_fraction: float = 0.8
    epochs: int = 400
    learning_rate: float = 1e-2
    momentum: float = 0.9

    def validate(self) -> None:
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if not 0 < self.train_fraction < 1:
            raise ValueError(
                "train_fraction must be in (0, 1)"
            )
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )


@dataclass(frozen=True)
class Normalization:
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
    ) -> "Normalization":
        mean = x.mean(
            axis=0
        ).astype(
            np.float64
        )
        std = x.std(
            axis=0
        ).astype(
            np.float64
        )
        std = np.where(
            std < 1e-8,
            1.0,
            std,
        )
        return cls(
            mean=mean.tolist(),
            std=std.tolist(),
        )

    def apply_np(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        return (
            x
            - np.asarray(
                self.mean,
                dtype=np.float32,
            )
        ) / np.asarray(
            self.std,
            dtype=np.float32,
        )

    def apply_tensor(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        mean = torch.tensor(
            self.mean,
            dtype=x.dtype,
            device=x.device,
        )
        std = torch.tensor(
            self.std,
            dtype=x.dtype,
            device=x.device,
        )
        return (x - mean) / std


def _load_records(
    paths: Iterable[Path],
) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with path.open(
            encoding="utf-8"
        ) as handle:
            for line in handle:
                if line.strip():
                    records.append(
                        json.loads(line)
                    )
    return records


def _arrays(
    records: list[dict],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    if not records:
        raise ValueError(
            "no predictor profiling records"
        )

    x = np.asarray(
        [
            [
                float(
                    row["token_block"]
                ),
                float(
                    row["active_blocks"]
                ),
                float(
                    row["gpu_util"]
                ),
            ]
            for row in records
        ],
        dtype=np.float32,
    )

    y = np.asarray(
        [
            float(
                row["attention_ms"]
            )
            for row in records
        ],
        dtype=np.float32,
    ).reshape(-1, 1)

    if not np.isfinite(x).all():
        raise ValueError(
            "non-finite predictor features"
        )
    if not np.isfinite(y).all():
        raise ValueError(
            "non-finite predictor targets"
        )
    if (y < 0).any():
        raise ValueError(
            "negative attention latency target"
        )

    return x, y


def train_predictor(
    *,
    records: list[dict],
    output: Path,
    config: PredictorTrainConfig,
    dense_ms: float,
    final_projection_ms: float,
) -> dict:
    config.validate()

    rng = random.Random(
        config.seed
    )
    records = list(records)
    rng.shuffle(records)

    if len(records) < config.samples:
        raise ValueError(
            "The SparKV paper trains the MLP on 6,000 samples. "
            f"Only {len(records)} profiling records are available; "
            f"need at least {config.samples}."
        )

    records = records[
        : config.samples
    ]

    x, y = _arrays(
        records
    )

    split = int(
        round(
            len(x)
            * config.train_fraction
        )
    )
    split = min(
        max(split, 1),
        len(x) - 1,
    )

    x_train_raw = x[:split]
    y_train = y[:split]
    x_test_raw = x[split:]
    y_test = y[split:]

    norm = Normalization.fit(
        x_train_raw
    )
    x_train = norm.apply_np(
        x_train_raw
    )
    x_test = norm.apply_np(
        x_test_raw
    )

    torch.manual_seed(
        config.seed
    )

    model = SparseAttentionMLP()
    criterion = nn.MSELoss()

    # The paper states SGD + MSE but does not disclose LR, momentum, or
    # epoch count.  These remain explicit experiment configuration rather
    # than silently claiming author-identical hyperparameters.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
    )

    tx = torch.from_numpy(
        x_train
    )
    ty = torch.from_numpy(
        y_train
    )

    for _ in range(
        config.epochs
    ):
        optimizer.zero_grad(
            set_to_none=True
        )
        pred = model(tx)
        loss = criterion(
            pred,
            ty,
        )
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        train_pred = (
            model(tx)
            .numpy()
            .reshape(-1)
        )
        test_pred = (
            model(
                torch.from_numpy(
                    x_test
                )
            )
            .numpy()
            .reshape(-1)
        )

    train_true = (
        y_train.reshape(-1)
    )
    test_true = (
        y_test.reshape(-1)
    )

    train_mae = float(
        np.mean(
            np.abs(
                train_pred
                - train_true
            )
        )
    )
    test_mae = float(
        np.mean(
            np.abs(
                test_pred
                - test_true
            )
        )
    )
    test_mape = float(
        np.mean(
            np.abs(
                test_pred
                - test_true
            )
            / np.clip(
                test_true,
                1e-6,
                None,
            )
        )
    )

    payload = {
        "state_dict":
            model.state_dict(),
        "normalization":
            asdict(norm),
        "feature_names":
            list(
                FEATURE_NAMES
            ),
        "dense_ms":
            float(dense_ms),
        "final_projection_ms":
            float(
                final_projection_ms
            ),
        "train_config":
            asdict(config),
        "metrics": {
            "train_mae_ms":
                train_mae,
            "test_mae_ms":
                test_mae,
            "test_mape":
                test_mape,
            "train_records":
                int(split),
            "test_records":
                int(
                    len(x)
                    - split
                ),
        },
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        payload,
        output,
    )

    return {
        "saved": str(output),
        **payload["metrics"],
        "dense_ms":
            float(dense_ms),
        "final_projection_ms":
            float(
                final_projection_ms
            ),
    }


class ComputationLatencyPredictor:
    def __init__(
        self,
        checkpoint: Path,
    ) -> None:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        self.model = (
            SparseAttentionMLP()
        )
        self.model.load_state_dict(
            payload[
                "state_dict"
            ]
        )
        self.model.eval()

        norm = payload[
            "normalization"
        ]
        self.normalization = (
            Normalization(
                mean=[
                    float(x)
                    for x in norm[
                        "mean"
                    ]
                ],
                std=[
                    float(x)
                    for x in norm[
                        "std"
                    ]
                ],
            )
        )

        self.dense_ms = float(
            payload["dense_ms"]
        )
        self.final_projection_ms = float(
            payload[
                "final_projection_ms"
            ]
        )

        self.metadata = payload

    def attention_ms(
        self,
        *,
        token_block: int,
        active_blocks: int,
        gpu_util: float,
    ) -> float:
        x = torch.tensor(
            [
                [
                    float(
                        token_block
                    ),
                    float(
                        active_blocks
                    ),
                    float(
                        gpu_util
                    ),
                ]
            ],
            dtype=torch.float32,
        )
        x = (
            self.normalization
            .apply_tensor(x)
        )

        with torch.no_grad():
            value = float(
                self.model(x)
                .item()
            )

        return max(
            1e-6,
            value,
        )

    def chunk_ms(
        self,
        *,
        token_block: int,
        active_blocks: int,
        gpu_util: float,
        final_layer: bool,
    ) -> float:
        if final_layer:
            return max(
                1e-6,
                self.final_projection_ms,
            )

        return (
            self.attention_ms(
                token_block=
                    token_block,
                active_blocks=
                    active_blocks,
                gpu_util=
                    gpu_util,
            )
            + self.dense_ms
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profiles",
        nargs="+",
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--dense-ms",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--final-projection-ms",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=6000,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    args = parser.parse_args()

    result = train_predictor(
        records=_load_records(
            [
                Path(x)
                for x in args.profiles
            ]
        ),
        output=Path(
            args.output
        ),
        config=PredictorTrainConfig(
            seed=args.seed,
            samples=args.samples,
            epochs=args.epochs,
            learning_rate=
                args.learning_rate,
            momentum=args.momentum,
        ),
        dense_ms=args.dense_ms,
        final_projection_ms=
            args.final_projection_ms,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
