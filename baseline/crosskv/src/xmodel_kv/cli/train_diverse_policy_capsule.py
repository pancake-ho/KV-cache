from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..diverse_handoff import load_manifest
from ..diverse_training import load_screen_rows, select_eligible_rows
from ..policy_capsule import (
    SoftPolicyCapsule,
    capsule_action_logits,
    start_capsule_readout,
)
from ..policy_imprinting import (
    build_chat_segments,
    greedy_action,
    parse_tool_call,
    prefill_legacy_cache,
    stitch_history_cache,
    validate_shared_handoff,
)
from .evaluate_diverse_controls import _appended_history_ids
from .evaluate_diverse_refresh import compact_target_refresh
from .screen_diverse_handoff import score_bfcl_call
from .train_static_read_lens import _cache_to


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a B-conditioned compact policy capsule on stale history KV"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--screen", required=True, nargs="+")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--train-rows", type=int, default=1200)
    parser.add_argument("--validation-rows", type=int, default=150)
    parser.add_argument("--test-rows", type=int, default=200)
    parser.add_argument("--capsule-tokens", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--selection-weight", type=float, default=10.0)
    parser.add_argument("--selection-window", type=int, default=12)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--contrast-margin", type=float, default=3.0)
    parser.add_argument("--drift-regularization", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    positive = (
        args.train_rows,
        args.validation_rows,
        args.test_rows,
        args.capsule_tokens,
        args.epochs,
        args.selection_window,
        args.max_new_tokens,
    )
    if min(positive) < 1:
        parser.error("row counts, token counts, epochs, and windows must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capsule = SoftPolicyCapsule(
        args.capsule_tokens, model.config.hidden_size
    ).to(device=args.device, dtype=torch.float32)
    capsule.initialize_from_text(model, tokenizer)

    manifest = load_manifest(Path(args.manifest))
    screens = load_screen_rows([Path(path) for path in args.screen])
    selected, selection_audit = select_eligible_rows(
        manifest,
        screens,
        limits={
            "train": args.train_rows,
            "validation": args.validation_rows,
            "iid_test": args.test_rows,
        },
    )
    (output_dir / "selection.json").write_text(
        json.dumps(selection_audit, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(selection_audit, ensure_ascii=False), flush=True)

    optimizer = AdamW(
        capsule.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best_score = None
    for epoch in range(1, args.epochs + 1):
        order = list(selected["train"])
        random.Random(args.seed + epoch).shuffle(order)
        capsule.train()
        totals = {"loss": 0.0, "ce": 0.0, "distill": 0.0, "contrast": 0.0}
        for step, row in enumerate(order, 1):
            optimizer.zero_grad(set_to_none=True)
            logits, action_ids, source_ids = _forward_training_example(
                model, tokenizer, capsule, row, device=args.device
            )
            losses = _training_losses(
                logits,
                action_ids,
                source_ids,
                row,
                capsule,
                selection_weight=args.selection_weight,
                selection_window=args.selection_window,
                distill_weight=args.distill_weight,
                contrast_weight=args.contrast_weight,
                contrast_margin=args.contrast_margin,
                drift_regularization=args.drift_regularization,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(capsule.parameters(), args.grad_clip)
            optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].detach())
            if step == 1 or step % 25 == 0:
                print(
                    {
                        "epoch": epoch,
                        "step": step,
                        "steps": len(order),
                        "mean_loss": totals["loss"] / step,
                        "mean_contrast": totals["contrast"] / step,
                    },
                    flush=True,
                )
            del logits, losses
            gc.collect()
            torch.cuda.empty_cache()
        capsule.eval()
        validation, _ = _evaluate(
            model,
            tokenizer,
            capsule,
            selected["validation"],
            max_new_tokens=args.max_new_tokens,
            collect_rows=False,
        )
        metrics = {
            "epoch": epoch,
            **{f"training_{name}": value / len(order) for name, value in totals.items()},
            "validation": validation,
            "capsule_norm": float(capsule.embeddings.float().norm().detach()),
            "capsule_drift": float(capsule.drift_regularization().detach()),
        }
        history.append(metrics)
        (output_dir / "metrics.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        score = (validation["target_accuracy"], -validation["source_accuracy"])
        checkpoint = {
            "state_dict": capsule.state_dict(),
            "capsule_tokens": capsule.tokens,
            "epoch": epoch,
            "metrics": metrics,
            "conditioning": "compact_target_text_then_shared_soft_suffix",
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if best_score is None or score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pt")

    checkpoint = torch.load(
        output_dir / "best.pt", map_location=args.device, weights_only=True
    )
    capsule.load_state_dict(checkpoint["state_dict"])
    capsule.eval()
    test_metrics, test_rows = _evaluate(
        model,
        tokenizer,
        capsule,
        selected["iid_test"],
        max_new_tokens=args.max_new_tokens,
        collect_rows=True,
    )
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "test_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in test_rows)
    )
    print(json.dumps({"final_test": test_metrics}, ensure_ascii=False), flush=True)


def _forward_training_example(model, tokenizer, capsule, row, *, device):
    source, target, tail_ids, readout_ids = _segments(tokenizer, row)
    source_cache = prefill_legacy_cache(
        model, source.prefix_ids + source.history_ids
    )
    target_prefix = prefill_legacy_cache(model, target.prefix_ids)
    stitched = stitch_history_cache(
        model,
        source_context_cache=source_cache,
        target_prefix_cache=target_prefix,
        source_prefix_length=len(source.prefix_ids),
        target_prefix_length=len(target.prefix_ids),
        transfer_history_length=len(target.history_ids),
    )
    cache = _cache_to(stitched, device)
    action_ids = tuple(row["screen"]["target_native"]["action_ids"])
    source_ids = tuple(
        tokenizer.encode(
            row["screen"]["source_native"]["text"], add_special_tokens=False
        )
    )
    logits = capsule_action_logits(
        model,
        cache,
        capsule,
        cached_tokens=len(target.prefix_ids) + len(target.history_ids),
        tail_ids=tail_ids,
        readout_ids=readout_ids,
        action_ids=action_ids,
    )
    del source_cache, target_prefix, stitched, cache
    return logits, action_ids, source_ids


def _training_losses(
    logits,
    action_ids,
    source_ids,
    row,
    capsule,
    *,
    selection_weight,
    selection_window,
    distill_weight,
    contrast_weight,
    contrast_margin,
    drift_regularization,
):
    device = logits.device
    labels = torch.tensor(action_ids, dtype=torch.long, device=device)
    token_ce = F.cross_entropy(
        logits.float().squeeze(0), labels, reduction="none"
    )
    difference = first_difference(action_ids, source_ids)
    weights = torch.ones_like(token_ce)
    if difference < len(action_ids):
        stop = min(len(action_ids), difference + selection_window)
        weights[difference:stop] = selection_weight
    ce = (token_ce * weights).sum() / weights.sum()

    topk_indices = torch.tensor(
        row["screen"]["target_native"]["teacher_topk_indices"],
        dtype=torch.long,
        device=device,
    )
    topk_log_probs = torch.tensor(
        row["screen"]["target_native"]["teacher_topk_log_probs"],
        dtype=torch.float32,
        device=device,
    )
    teacher = topk_log_probs.softmax(dim=-1)
    student = logits.float().log_softmax(dim=-1).squeeze(0).gather(
        dim=-1, index=topk_indices
    )
    token_distill = -(teacher * student).sum(dim=-1)
    distill = (token_distill * weights).sum() / weights.sum()

    contrast = logits.new_zeros((), dtype=torch.float32)
    if difference < min(len(action_ids), len(source_ids)):
        target_logit = logits[0, difference, action_ids[difference]].float()
        source_logit = logits[0, difference, source_ids[difference]].float()
        contrast = F.softplus(contrast_margin - (target_logit - source_logit))
    loss = (
        ce
        + distill_weight * distill
        + contrast_weight * contrast
        + drift_regularization * capsule.drift_regularization()
    )
    return {"loss": loss, "ce": ce, "distill": distill, "contrast": contrast}


@torch.inference_mode()
def _evaluate(model, tokenizer, capsule, rows, *, max_new_tokens, collect_rows):
    target_correct = source_correct = valid_json = 0
    details = []
    for index, row in enumerate(rows, 1):
        source, target, tail_ids, readout_ids = _segments(tokenizer, row)
        source_cache = prefill_legacy_cache(
            model, source.prefix_ids + source.history_ids
        )
        target_prefix = prefill_legacy_cache(model, target.prefix_ids)
        stitched = stitch_history_cache(
            model,
            source_context_cache=source_cache,
            target_prefix_cache=target_prefix,
            source_prefix_length=len(source.prefix_ids),
            target_prefix_length=len(target.prefix_ids),
            transfer_history_length=len(target.history_ids),
        )
        factory = lambda: start_capsule_readout(
            model,
            legacy_cache=stitched,
            capsule=capsule,
            cached_tokens=len(target.prefix_ids) + len(target.history_ids),
            tail_ids=tail_ids,
            readout_ids=readout_ids,
        )
        _, text = greedy_action(
            model, factory, tokenizer, max_new_tokens=max_new_tokens
        )
        call = parse_tool_call(text)
        target_score = score_bfcl_call(
            call,
            expected=row["target_expected"],
            argument_options=row["target_argument_options"],
        )
        source_score = score_bfcl_call(
            call,
            expected=row["source_expected"],
            argument_options=row["source_argument_options"],
        )
        target_correct += int(target_score["compatible_em"])
        source_correct += int(source_score["compatible_em"])
        valid_json += int(call.valid_json)
        if collect_rows:
            details.append(
                {
                    "id": row["id"],
                    "target": target_score,
                    "source": source_score,
                    "generation": text,
                }
            )
        if index % 25 == 0:
            print(
                {
                    "evaluation": index,
                    "rows": len(rows),
                    "target": target_correct,
                    "source": source_correct,
                },
                flush=True,
            )
        del source_cache, target_prefix, stitched
        gc.collect()
        torch.cuda.empty_cache()
    count = len(rows)
    return (
        {
            "rows": count,
            "target_correct": target_correct,
            "target_accuracy": target_correct / count,
            "source_correct": source_correct,
            "source_accuracy": source_correct / count,
            "valid_json": valid_json,
            "valid_json_rate": valid_json / count,
        },
        details,
    )


def _segments(tokenizer, row):
    source = build_chat_segments(
        tokenizer,
        system_prompt=row["source_prompt"],
        tools=row["source_tools"],
        history=row["history"],
    )
    target = build_chat_segments(
        tokenizer,
        system_prompt=row["target_prompt"],
        tools=row["target_tools"],
        history=row["history"],
    )
    validate_shared_handoff(source, target)
    appended = build_chat_segments(
        tokenizer,
        system_prompt=row["target_prompt"],
        tools=row["target_tools"],
        history=[
            *row["history"],
            {"role": "system", "content": compact_target_refresh(row)},
        ],
    )
    return source, target, _appended_history_ids(target, appended), appended.readout_ids


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


if __name__ == "__main__":
    main()
