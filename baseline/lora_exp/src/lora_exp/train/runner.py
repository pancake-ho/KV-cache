from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainingArguments

from lora_exp.common.seed import set_seed
from lora_exp.model.factory import (
    load_base_model,
    load_tokenizer,
)
from lora_exp.model.lora_factory import (
    attach_lora,
)
from lora_exp.model.validate import (
    get_parameter_statistics,
    validate_lora_only_trainable,
)

from lora_exp.train.packing import (
    FixedBlockCausalCollator,
)


def _save_json(
    payload: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def train_one_adapter(
    *,
    dataset_name: str,
    train_dataset,
    eval_dataset,
    model_cfg: dict[str, Any],
    lora_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    output_dir: str | Path,
    seed: int,
) -> dict[str, Any]:
    """
    Train one independent LoRA adapter from a freshly loaded,
    identical frozen Qwen3-4B base model.

    Important:
      - base model is reloaded for every adapter
      - no adapter is merged
      - no 4/8-bit model quantization
      - each dataset receives exactly the same packed-token budget
    """
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_adapter_dir = (
        output_dir / "adapter_final"
    )

    summary_path = (
        output_dir
        / "train_summary.json"
    )

    if (
        train_cfg.get(
            "resume_completed",
            True,
        )
        and final_adapter_dir.exists()
        and summary_path.exists()
    ):
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            old_summary = json.load(f)

        if old_summary.get("status") == "PASS":
            print(
                f"[SKIP] {dataset_name}: "
                "completed adapter already exists."
            )

            return old_summary

    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. "
            "CPU fallback is not allowed."
        )

    print(
        f"\n{'=' * 80}\n"
        f"TRAIN ADAPTER: {dataset_name}\n"
        f"{'=' * 80}"
    )

    tokenizer = load_tokenizer(
        model_cfg
    )

    tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        "[MODEL] Loading fresh "
        "Qwen3-4B BF16 base..."
    )

    model = load_base_model(
        model_cfg,
        device="cuda",
        use_cache=False,
    )

    # Freeze base explicitly.
    for parameter in model.parameters():
        parameter.requires_grad = False

    model = attach_lora(
        model,
        lora_cfg,
    )

    model.config.use_cache = False

    validate_lora_only_trainable(
        model
    )

    parameter_stats = (
        get_parameter_statistics(
            model
        )
    )

    model.print_trainable_parameters()

    if train_cfg.get(
        "gradient_checkpointing",
        False,
    ):
        print(
            "[TRAIN] Enabling gradient checkpointing"
        )

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )

        # Required for some frozen-base PEFT +
        # checkpointing combinations.
        if hasattr(
            model,
            "enable_input_require_grads",
        ):
            model.enable_input_require_grads()

    checkpoints_dir = (
        output_dir / "checkpoints"
    )

    training_args = TrainingArguments(
        output_dir=str(
            checkpoints_dir
        ),

        per_device_train_batch_size=int(
            train_cfg[
                "per_device_train_batch_size"
            ]
        ),

        per_device_eval_batch_size=int(
            train_cfg[
                "per_device_eval_batch_size"
            ]
        ),

        gradient_accumulation_steps=int(
            train_cfg[
                "gradient_accumulation_steps"
            ]
        ),

        num_train_epochs=float(
            train_cfg[
                "num_train_epochs"
            ]
        ),

        learning_rate=float(
            train_cfg[
                "learning_rate"
            ]
        ),

        weight_decay=float(
            train_cfg[
                "weight_decay"
            ]
        ),

        warmup_ratio=float(
            train_cfg[
                "warmup_ratio"
            ]
        ),

        max_grad_norm=float(
            train_cfg[
                "max_grad_norm"
            ]
        ),

        optim=str(
            train_cfg["optim"]
        ),

        lr_scheduler_type=str(
            train_cfg[
                "lr_scheduler_type"
            ]
        ),

        bf16=bool(
            train_cfg["bf16"]
        ),

        fp16=bool(
            train_cfg["fp16"]
        ),

        gradient_checkpointing=bool(
            train_cfg[
                "gradient_checkpointing"
            ]
        ),

        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_strategy="steps",
        logging_steps=int(
            train_cfg["logging_steps"]
        ),

        save_total_limit=int(
            train_cfg[
                "save_total_limit"
            ]
        ),

        auto_find_batch_size=bool(
            train_cfg[
                "auto_find_batch_size"
            ]
        ),

        report_to="none",

        remove_unused_columns=False,

        seed=int(seed),
        data_seed=int(seed),

        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        save_safetensors=True,

        # Representation study:
        # avoid hidden TF32 differences.
        tf32=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=(
            FixedBlockCausalCollator()
        ),
    )

    train_result = trainer.train()

    eval_metrics = trainer.evaluate()

    if (
        not torch.isfinite(
            torch.tensor(
                float(
                    eval_metrics[
                        "eval_loss"
                    ]
                )
            )
        )
    ):
        raise RuntimeError(
            f"{dataset_name}: "
            "non-finite eval loss"
        )

    print(
        "[SAVE] Saving independent LoRA adapter..."
    )

    final_adapter_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        final_adapter_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        output_dir / "tokenizer"
    )

    trainer.state.save_to_json(
        str(
            output_dir
            / "trainer_state.json"
        )
    )

    summary = {
        "status": "PASS",
        "dataset": dataset_name,
        "seed": int(seed),

        "train_blocks": int(
            len(train_dataset)
        ),

        "eval_blocks": int(
            len(eval_dataset)
        ),

        "sequence_length": int(
            train_dataset[0][
                "input_ids"
            ].__len__()
        ),

        "train_tokens_per_epoch": int(
            len(train_dataset)
            * len(
                train_dataset[0][
                    "input_ids"
                ]
            )
        ),

        "parameter_statistics": (
            parameter_stats
        ),

        "train_metrics": dict(
            train_result.metrics
        ),

        "eval_metrics": dict(
            eval_metrics
        ),

        "best_model_checkpoint": (
            trainer.state.best_model_checkpoint
        ),

        "best_metric": (
            trainer.state.best_metric
        ),

        "adapter_dir": str(
            final_adapter_dir
        ),
    }

    _save_json(
        summary,
        summary_path,
    )

    print(
        f"[PASS] {dataset_name}"
    )

    # Free the 4B model before loading the next adapter.
    del trainer
    del model

    gc.collect()
    torch.cuda.empty_cache()

    return summary