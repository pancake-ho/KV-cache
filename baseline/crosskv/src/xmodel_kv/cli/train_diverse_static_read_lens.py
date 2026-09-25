from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..diverse_handoff import DEFAULT_SPLIT_SIZES, load_manifest
from ..diverse_training import load_screen_rows, select_eligible_rows
from ..policy_imprinting import (
    build_chat_segments,
    greedy_action,
    parse_tool_call,
    prefill_legacy_cache,
    start_readout,
    stitch_history_cache,
    validate_shared_handoff,
)
from ..read_lens import (
    StaticReadLensBank,
    lens_norms,
    lens_regularization,
    register_read_lens_attention,
)
from .screen_diverse_handoff import score_bfcl_call
from .train_static_read_lens import _cache_to, _parse_layers, _student_action_logits


EVAL_SPLITS = (
    "validation", "iid_test", "pair_ood", "policy_ood", "tool_ood",
    "prompt_ood", "length_ood",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a diverse static Agent-KV read lens with synchronous data parallelism"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--screen", required=True, nargs="+")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--train-rows", type=int, default=1002)
    parser.add_argument("--validation-rows", type=int, default=150)
    parser.add_argument("--iid-test-rows", type=int, default=250)
    parser.add_argument("--pair-ood-rows", type=int, default=200)
    parser.add_argument("--policy-ood-rows", type=int, default=200)
    parser.add_argument("--tool-ood-rows", type=int, default=200)
    parser.add_argument("--prompt-ood-rows", type=int, default=200)
    parser.add_argument("--length-ood-rows", type=int, default=100)
    parser.add_argument(
        "--test-splits",
        default=",".join(EVAL_SPLITS[1:]),
        help="Comma-separated final test splits; validation is always used for selection",
    )
    parser.add_argument("--layers", default="18-29")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--eval-every", type=int, default=1)
    args = parser.parse_args()
    test_splits = tuple(
        part.strip() for part in args.test_splits.split(",") if part.strip()
    )
    unknown_test_splits = set(test_splits) - set(EVAL_SPLITS[1:])
    if not test_splits or unknown_test_splits:
        parser.error(
            f"test-splits must select from {EVAL_SPLITS[1:]}; "
            f"unknown={sorted(unknown_test_splits)}"
        )
    row_limits = {
        "iid_test": args.iid_test_rows,
        "pair_ood": args.pair_ood_rows,
        "policy_ood": args.policy_ood_rows,
        "tool_ood": args.tool_ood_rows,
        "prompt_ood": args.prompt_ood_rows,
        "length_ood": args.length_ood_rows,
    }
    if min(
        args.train_rows,
        args.validation_rows,
        *(row_limits[split] for split in test_splits),
        args.rank,
        args.epochs,
        args.max_new_tokens,
        args.eval_every,
    ) < 1:
        parser.error("row counts, rank, epochs, and evaluation interval must be positive")

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        global_rank, world_size, device = 0, 1, args.device
    is_main = global_rank == 0
    if args.train_rows % world_size:
        raise SystemExit("train-rows must be divisible by the distributed world size")
    layers = _parse_layers(args.layers)
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    register_read_lens_attention()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map={"": device},
        attn_implementation="static_read_lens",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bank = StaticReadLensBank(model, layers=layers, rank=args.rank).to(
        device=device, dtype=torch.float32
    )
    # All ranks have identical initialization; gradients are averaged manually.
    manifest = load_manifest(Path(args.manifest))
    screens = load_screen_rows([Path(path) for path in args.screen])
    selected, selection_audit = select_eligible_rows(
        manifest,
        screens,
        limits={
            "train": args.train_rows,
            "validation": args.validation_rows,
            **{split: row_limits[split] for split in test_splits},
        },
    )
    if is_main:
        (output_dir / "selection.json").write_text(
            json.dumps(selection_audit, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(selection_audit, ensure_ascii=False), flush=True)

    optimizer = AdamW(
        bank.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best_score = None
    for epoch in range(1, args.epochs + 1):
        order = list(selected["train"])
        random.Random(args.seed + epoch).shuffle(order)
        local_rows = order[global_rank::world_size]
        bank.train()
        local_loss = 0.0
        local_ce = 0.0
        local_distill = 0.0
        for step, row in enumerate(local_rows, 1):
            optimizer.zero_grad(set_to_none=True)
            logits, action_ids = _forward_training_example(
                model, tokenizer, bank, row, device=device
            )
            labels = torch.tensor(action_ids, dtype=torch.long, device=device)
            ce = F.cross_entropy(logits.float().squeeze(0), labels)
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
            distill = -(teacher * student).sum(dim=-1).mean()
            regularizer = lens_regularization(bank)
            loss = distill + args.ce_weight * ce + args.regularization * regularizer
            loss.backward()
            if distributed:
                _sync_gradients(bank, world_size)
            torch.nn.utils.clip_grad_norm_(bank.parameters(), args.grad_clip)
            optimizer.step()
            local_loss += float(loss.detach())
            local_ce += float(ce.detach())
            local_distill += float(distill.detach())
            del logits, labels, topk_indices, topk_log_probs, teacher, student, loss
            if is_main and (step == 1 or step % 25 == 0):
                print(
                    {
                        "epoch": epoch,
                        "step": step,
                        "steps_per_rank": len(local_rows),
                        "mean_loss": local_loss / step,
                    },
                    flush=True,
                )
        totals = torch.tensor(
            [local_loss, local_ce, local_distill, len(local_rows)],
            device=device,
            dtype=torch.float64,
        )
        if distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        metrics = {
            "epoch": epoch,
            "training_loss": totals[0].item() / totals[3].item(),
            "training_ce": totals[1].item() / totals[3].item(),
            "training_distill": totals[2].item() / totals[3].item(),
        }
        if epoch % args.eval_every == 0:
            bank.eval()
            metrics["validation"] = _distributed_evaluate(
                model,
                tokenizer,
                bank,
                selected["validation"],
                global_rank=global_rank,
                world_size=world_size,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
        if is_main:
            metrics["lens_norms"] = lens_norms(bank)
            history.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            (output_dir / "metrics.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2) + "\n"
            )
            validation = metrics.get("validation", {})
            score = (
                validation.get("target_accuracy", -1.0),
                -validation.get("source_accuracy", 1.0),
            )
            if best_score is None or score > best_score:
                best_score = score
                _save_checkpoint(
                    output_dir / "best.pt", bank, layers, args.rank, epoch, metrics
                )
            _save_checkpoint(
                output_dir / "last.pt", bank, layers, args.rank, epoch, metrics
            )
        if distributed:
            dist.barrier()
    if distributed:
        dist.barrier()
    checkpoint = torch.load(
        output_dir / "best.pt", map_location=device, weights_only=True
    )
    bank.load_state_dict(checkpoint["state_dict"])
    bank.eval()
    test_metrics = {}
    for split in test_splits:
        test_metrics[split] = _distributed_evaluate(
            model,
            tokenizer,
            bank,
            selected[split],
            global_rank=global_rank,
            world_size=world_size,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
    if is_main:
        (output_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps({"final_test": test_metrics}, ensure_ascii=False), flush=True)
    if distributed:
        dist.destroy_process_group()


def _forward_training_example(model, tokenizer, bank, row, *, device):
    bank.disable(model)
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
    source_cache = prefill_legacy_cache(model, source.prefix_ids + source.history_ids)
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
    del source_cache, target_prefix, stitched
    action_ids = tuple(row["screen"]["target_native"]["action_ids"])
    bank.configure(
        model,
        stale_start=len(target.prefix_ids),
        stale_end=len(target.prefix_ids) + len(target.history_ids),
        enabled=True,
    )
    logits = _student_action_logits(
        model,
        cache,
        cached_tokens=len(target.prefix_ids) + len(target.history_ids),
        readout_ids=target.readout_ids,
        action_ids=action_ids,
    )
    del cache
    return logits, action_ids


@torch.inference_mode()
def _evaluate_one(model, tokenizer, bank, row, *, max_new_tokens):
    bank.disable(model)
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
    source_cache = prefill_legacy_cache(model, source.prefix_ids + source.history_ids)
    target_prefix = prefill_legacy_cache(model, target.prefix_ids)
    stitched = stitch_history_cache(
        model,
        source_context_cache=source_cache,
        target_prefix_cache=target_prefix,
        source_prefix_length=len(source.prefix_ids),
        target_prefix_length=len(target.prefix_ids),
        transfer_history_length=len(target.history_ids),
    )
    bank.configure(
        model,
        stale_start=len(target.prefix_ids),
        stale_end=len(target.prefix_ids) + len(target.history_ids),
        enabled=True,
    )
    factory = lambda: start_readout(
        model,
        legacy_cache=stitched,
        cached_tokens=len(target.prefix_ids) + len(target.history_ids),
        input_ids=target.readout_ids,
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
    del source_cache, target_prefix, stitched
    return int(target_score["compatible_em"]), int(source_score["compatible_em"])


def _distributed_evaluate(
    model,
    tokenizer,
    bank,
    rows,
    *,
    global_rank,
    world_size,
    device,
    max_new_tokens,
):
    target_correct = 0
    source_correct = 0
    local_rows = rows[global_rank::world_size]
    for row in local_rows:
        target, source = _evaluate_one(
            model, tokenizer, bank, row, max_new_tokens=max_new_tokens
        )
        target_correct += target
        source_correct += source
        gc.collect()
        torch.cuda.empty_cache()
    counts = torch.tensor(
        [target_correct, source_correct, len(local_rows)],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return {
        "rows": int(counts[2].item()),
        "target_accuracy": counts[0].item() / counts[2].item(),
        "source_accuracy": counts[1].item() / counts[2].item(),
    }


def _save_checkpoint(path, bank, layers, rank, epoch, metrics):
    torch.save(
        {
            "state_dict": bank.state_dict(),
            "layers": layers,
            "rank": rank,
            "epoch": epoch,
            "metrics": metrics,
            "conditioning": "static",
        },
        path,
    )


def _sync_gradients(bank, world_size):
    parameters = [parameter for parameter in bank.parameters() if parameter.grad is not None]
    flat = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    offset = 0
    for parameter in parameters:
        elements = parameter.numel()
        parameter.grad.copy_(flat[offset : offset + elements].view_as(parameter))
        offset += elements


if __name__ == "__main__":
    main()
