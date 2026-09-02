from pathlib import Path

path = Path("baseline/sparkv/overhead_model.py")
text = path.read_text(encoding="utf-8")

def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"[ERROR] {label}: expected exactly one match, found {count}. "
            "Refusing to modify a different/local-newer file."
        )
    text = text.replace(old, new, 1)

replace_once(
'''    x_train_raw = x[:split]
    y_train = y[:split]
    x_test_raw = x[split:]
    y_test = y[split:]

    norm = Normalization.fit(
''',
'''    x_train_raw = x[:split]
    y_train = y[:split]
    x_test_raw = x[split:]
    y_test = y[split:]

    # Numerical conditioning for the scalar latency target.
    # The paper fixes the 3->48->24->1 MLP, SGD, MSE and 6000/80:20 setup,
    # but does not disclose optimizer hyperparameters. Standardizing y keeps
    # the same one-output MLP and changes MSE only by a positive scale factor.
    target_mean = float(
        y_train.mean()
    )
    target_std = float(
        y_train.std()
    )
    if target_std < 1e-8:
        target_std = 1.0

    y_train_normalized = (
        (
            y_train
            - target_mean
        )
        / target_std
    ).astype(
        np.float32
    )

    norm = Normalization.fit(
''',
"insert target normalization",
)

replace_once(
'''    ty = torch.from_numpy(
        y_train
    )
''',
'''    ty = torch.from_numpy(
        y_train_normalized
    )
''',
"use normalized target",
)

replace_once(
'''    with torch.no_grad():
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
''',
'''    with torch.no_grad():
        train_pred_normalized = (
            model(tx)
            .numpy()
            .reshape(-1)
        )
        test_pred_normalized = (
            model(
                torch.from_numpy(
                    x_test
                )
            )
            .numpy()
            .reshape(-1)
        )

    train_pred = (
        train_pred_normalized
        * target_std
        + target_mean
    )
    test_pred = (
        test_pred_normalized
        * target_std
        + target_mean
    )

    train_true = (
''',
"unnormalize train/test predictions",
)

replace_once(
'''        "normalization":
            asdict(norm),
        "feature_names":
''',
'''        "normalization":
            asdict(norm),
        "target_normalization": {
            "mean":
                target_mean,
            "std":
                target_std,
        },
        "feature_names":
''',
"save target normalization",
)

replace_once(
'''        self.dense_ms = float(
            payload["dense_ms"]
        )
''',
'''        target_norm = payload.get(
            "target_normalization",
            {
                "mean": 0.0,
                "std": 1.0,
            },
        )
        self.target_mean = float(
            target_norm[
                "mean"
            ]
        )
        self.target_std = float(
            target_norm[
                "std"
            ]
        )

        self.dense_ms = float(
            payload["dense_ms"]
        )
''',
"load target normalization",
)

replace_once(
'''        with torch.no_grad():
            value = float(
                self.model(x)
                .item()
            )

        return max(
            1e-6,
            value,
        )
''',
'''        with torch.no_grad():
            normalized_value = float(
                self.model(x)
                .item()
            )

        value = (
            normalized_value
            * self.target_std
            + self.target_mean
        )

        return max(
            1e-6,
            value,
        )
''',
"unnormalize inference output",
)

path.write_text(text, encoding="utf-8")
print("[OK] patched", path)
