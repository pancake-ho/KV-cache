from __future__ import annotations

import argparse
import gc
import json
import random
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..diverse_handoff import load_manifest
from ..diverse_training import load_screen_rows, select_eligible_rows
from ..policy_capsule import (
    PromptPolicyCompiler,
    compiled_capsule_action_logits,
    start_compiled_capsule_readout,
    start_embedding_capsule_readout,
)
from ..policy_imprinting import (
    build_chat_segments,
    greedy_action,
    parse_tool_call,
    prefill_legacy_cache,
    start_readout,
    stitch_history_cache,
    validate_shared_handoff,
)
from .evaluate_diverse_controls import NEUTRAL_RECEIVER_PROMPT
from .screen_diverse_handoff import score_bfcl_call
from .train_diverse_policy_capsule import first_difference
from .train_static_read_lens import _cache_to


TEST_SPLITS = (
    "iid_test",
    "pair_ood",
    "policy_ood",
    "tool_ood",
    "prompt_ood",
    "length_ood",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a receiver system prompt into fixed-length soft tail slots "
            "without appending row-specific plaintext after the reused history"
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--screen", required=True, nargs="+")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--train-rows", type=int, default=1200)
    parser.add_argument(
        "--train-repeat",
        type=int,
        default=1,
        help="repeat the selected training rows inside each epoch (diagnostic use)",
    )
    parser.add_argument(
        "--diagnostic-train-eval-rows",
        type=int,
        default=0,
        help="also generate on this many training rows after each epoch",
    )
    parser.add_argument("--validation-rows", type=int, default=150)
    parser.add_argument("--test-rows", type=int, default=200)
    parser.add_argument("--length-test-rows", type=int, default=100)
    parser.add_argument(
        "--test-splits", default=",".join(TEST_SPLITS), help="comma-separated splits"
    )
    parser.add_argument("--capsule-tokens", type=int, default=16)
    parser.add_argument("--compiler-dim", type=int, default=256)
    parser.add_argument("--compiler-layers", type=int, default=2)
    parser.add_argument("--compiler-heads", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument(
        "--prompt-representation",
        choices=("input_embeddings", "last_hidden"),
        default="last_hidden",
        help=(
            "last_hidden uses a frozen, amortizable base-model prompt encoding; "
            "input_embeddings is the language-untrained ablation"
        ),
    )
    parser.add_argument(
        "--compiler-input",
        choices=("full_prompt", "binding"),
        default="full_prompt",
        help=(
            "full_prompt is the proposed semantic compiler; binding is an oracle "
            "policy=value descriptor used to localize semantic-extraction failures"
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--selection-weight", type=float, default=10.0)
    parser.add_argument("--selection-window", type=int, default=12)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--contrast-margin", type=float, default=3.0)
    parser.add_argument(
        "--conditioning-weight",
        type=float,
        default=1.0,
        help=(
            "counterfactual source-prompt supervision at the first target/source "
            "action divergence"
        ),
    )
    parser.add_argument("--drift-regularization", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    test_splits = _parse_test_splits(parser, args.test_splits)
    positive = (
        args.train_rows,
        args.train_repeat,
        args.validation_rows,
        args.test_rows,
        args.length_test_rows,
        args.capsule_tokens,
        args.compiler_dim,
        args.compiler_layers,
        args.compiler_heads,
        args.max_prompt_tokens,
        args.epochs,
        args.selection_window,
        args.max_new_tokens,
    )
    if min(positive) < 1:
        parser.error("row counts, dimensions, epochs, and windows must be positive")
    if args.dropout < 0:
        parser.error("dropout must be non-negative")
    if args.diagnostic_train_eval_rows < 0:
        parser.error("diagnostic-train-eval-rows must be non-negative")
    if args.eval_only and not args.checkpoint:
        parser.error("--eval-only requires --checkpoint")

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
    compiler = PromptPolicyCompiler(
        tokens=args.capsule_tokens,
        model_hidden_size=model.config.hidden_size,
        compiler_dim=args.compiler_dim,
        layers=args.compiler_layers,
        heads=args.compiler_heads,
        max_prompt_tokens=args.max_prompt_tokens,
        dropout=args.dropout,
    ).to(device=args.device, dtype=torch.float32)
    compiler.initialize_from_text(model, tokenizer)

    manifest = load_manifest(Path(args.manifest))
    screens = load_screen_rows([Path(path) for path in args.screen])
    limits = {
        **({} if args.eval_only else {
            "train": args.train_rows,
            "validation": args.validation_rows,
        }),
        **{
            split: (
                args.length_test_rows if split == "length_ood" else args.test_rows
            )
            for split in test_splits
        },
    }
    selected, selection_audit = select_eligible_rows(
        manifest, screens, limits=limits
    )
    prompt_lengths = _prompt_length_audit(
        tokenizer, selected, compiler_input=args.compiler_input
    )
    if prompt_lengths["max"] > args.max_prompt_tokens:
        raise ValueError(
            f"selected prompt length {prompt_lengths['max']} exceeds "
            f"--max-prompt-tokens={args.max_prompt_tokens}"
        )
    (output_dir / "selection.json").write_text(
        json.dumps(selection_audit, ensure_ascii=False, indent=2) + "\n"
    )
    config = {**vars(args), "parsed_test_splits": list(test_splits)}
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"selection": selection_audit, "prompt_lengths": prompt_lengths},
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.checkpoint:
        checkpoint = torch.load(
            args.checkpoint, map_location=args.device, weights_only=True
        )
        compiler.load_state_dict(checkpoint["state_dict"])
        compiler.eval()
    if args.eval_only:
        _evaluate_splits(
            model,
            tokenizer,
            compiler,
            selected,
            test_splits,
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            prompt_representation=args.prompt_representation,
            compiler_input=args.compiler_input,
        )
        return

    optimizer = AdamW(
        compiler.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best_score = None
    for epoch in range(1, args.epochs + 1):
        order = list(selected["train"]) * args.train_repeat
        random.Random(args.seed + epoch).shuffle(order)
        compiler.train()
        totals = {
            "loss": 0.0,
            "ce": 0.0,
            "distill": 0.0,
            "contrast": 0.0,
            "conditioning": 0.0,
            "drift": 0.0,
        }
        for step, row in enumerate(order, 1):
            optimizer.zero_grad(set_to_none=True)
            (
                logits,
                compiled,
                source_prompt_logits,
                source_compiled,
                action_ids,
                source_ids,
            ) = _forward_training_example(
                model,
                tokenizer,
                compiler,
                row,
                device=args.device,
                require_source_prompt=args.conditioning_weight > 0,
                prompt_representation=args.prompt_representation,
                compiler_input=args.compiler_input,
            )
            losses = _training_losses(
                logits,
                compiled,
                source_prompt_logits,
                source_compiled,
                action_ids,
                source_ids,
                row,
                compiler,
                selection_weight=args.selection_weight,
                selection_window=args.selection_window,
                distill_weight=args.distill_weight,
                contrast_weight=args.contrast_weight,
                contrast_margin=args.contrast_margin,
                conditioning_weight=args.conditioning_weight,
                drift_regularization=args.drift_regularization,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(compiler.parameters(), args.grad_clip)
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
                        "mean_conditioning": totals["conditioning"] / step,
                    },
                    flush=True,
                )
            del logits, compiled, source_prompt_logits, source_compiled, losses
            gc.collect()
            torch.cuda.empty_cache()
        compiler.eval()
        validation, _ = _evaluate(
            model,
            tokenizer,
            compiler,
            selected["validation"],
            max_new_tokens=args.max_new_tokens,
            collect_rows=False,
            prompt_representation=args.prompt_representation,
            compiler_input=args.compiler_input,
        )
        train_diagnostic = None
        if args.diagnostic_train_eval_rows:
            train_diagnostic, _ = _evaluate(
                model,
                tokenizer,
                compiler,
                selected["train"][: args.diagnostic_train_eval_rows],
                max_new_tokens=args.max_new_tokens,
                collect_rows=False,
                prompt_representation=args.prompt_representation,
                compiler_input=args.compiler_input,
            )
        metrics = {
            "epoch": epoch,
            **{
                f"training_{name}": value / len(order)
                for name, value in totals.items()
            },
            "validation": validation,
            "train_diagnostic": train_diagnostic,
            "compiler_parameters": sum(
                parameter.numel() for parameter in compiler.parameters()
            ),
            "compiler_parameter_norm": _parameter_norm(compiler),
        }
        history.append(metrics)
        (output_dir / "metrics.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        target = validation["target_prompt"]
        swapped = validation["source_prompt"]
        score = (
            target["target_accuracy"],
            target["target_accuracy"] - swapped["target_accuracy"],
            -target["source_accuracy"],
        )
        checkpoint = {
            "state_dict": compiler.state_dict(),
            "capsule_tokens": compiler.tokens,
            "epoch": epoch,
            "metrics": metrics,
            "conditioning": "target_system_prompt_to_soft_tail_no_plaintext_tail",
            "prompt_representation": args.prompt_representation,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if best_score is None or score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pt")

    checkpoint = torch.load(
        output_dir / "best.pt", map_location=args.device, weights_only=True
    )
    compiler.load_state_dict(checkpoint["state_dict"])
    compiler.eval()
    _evaluate_splits(
        model,
        tokenizer,
        compiler,
        selected,
        test_splits,
        output_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        prompt_representation=args.prompt_representation,
        compiler_input=args.compiler_input,
    )


def _parse_test_splits(parser, value: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = sorted(set(result) - set(TEST_SPLITS))
    if not result or unknown:
        parser.error(f"test-splits must be non-empty members of {TEST_SPLITS}; unknown={unknown}")
    return result


def _compiler_prompt(row: dict, side: str, compiler_input: str) -> str:
    if side not in {"source", "target"}:
        raise ValueError(f"unsupported compiler prompt side: {side}")
    if compiler_input == "full_prompt":
        return row[f"{side}_prompt"]
    if compiler_input == "binding":
        return f"{row[f'{side}_policy']}={row[f'{side}_policy_value']}"
    raise ValueError(f"unsupported compiler input: {compiler_input}")


def _evaluate_splits(
    model,
    tokenizer,
    compiler,
    selected,
    test_splits,
    *,
    output_dir,
    max_new_tokens,
    prompt_representation,
    compiler_input,
):
    all_metrics = {}
    for split in test_splits:
        split_metrics, split_rows = _evaluate(
            model,
            tokenizer,
            compiler,
            selected[split],
            max_new_tokens=max_new_tokens,
            collect_rows=True,
            prompt_representation=prompt_representation,
            compiler_input=compiler_input,
        )
        all_metrics[split] = split_metrics
        (output_dir / f"test_rows.{split}.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows
            )
        )
        print(
            json.dumps({"final_test": {split: split_metrics}}, ensure_ascii=False),
            flush=True,
        )
    (output_dir / "test_metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n"
    )
    return all_metrics


def _prompt_length_audit(
    tokenizer, selected, *, compiler_input="full_prompt"
) -> dict[str, int | float]:
    lengths = [
        len(
            tokenizer.encode(
                _compiler_prompt(row, "target", compiler_input),
                add_special_tokens=False,
            )
        )
        for rows in selected.values()
        for row in rows
    ]
    ordered = sorted(lengths)
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }


def _parameter_norm(module) -> float:
    total = sum(
        parameter.detach().float().square().sum()
        for parameter in module.parameters()
    )
    return float(total.sqrt())


@torch.no_grad()
def _encode_prompt_states(model, prompt_ids, *, representation):
    if not prompt_ids:
        raise ValueError("prompt IDs must be non-empty")
    device = model.model.embed_tokens.weight.device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if representation == "input_embeddings":
        return model.model.embed_tokens(ids).squeeze(0).float()
    if representation != "last_hidden":
        raise ValueError(f"unsupported prompt representation: {representation}")
    attention_mask = torch.ones_like(ids)
    output = model.model(
        input_ids=ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    return output.last_hidden_state.squeeze(0).float()


def _forward_training_example(
    model,
    tokenizer,
    compiler,
    row,
    *,
    device,
    require_source_prompt,
    prompt_representation,
    compiler_input="full_prompt",
):
    _, neutral, stitched, cached_tokens = _student_cache(
        model, tokenizer, row, device=device
    )
    action_ids = tuple(row["screen"]["target_native"]["action_ids"])
    source_ids = tuple(
        tokenizer.encode(
            row["screen"]["source_native"]["text"], add_special_tokens=False
        )
    )
    prompt_ids = tuple(
        tokenizer.encode(
            _compiler_prompt(row, "target", compiler_input),
            add_special_tokens=False,
        )
    )
    prompt_states = _encode_prompt_states(
        model, prompt_ids, representation=prompt_representation
    )
    logits, compiled = compiled_capsule_action_logits(
        model,
        stitched,
        compiler,
        cached_tokens=cached_tokens,
        prompt_states=prompt_states,
        readout_ids=neutral.readout_ids,
        action_ids=action_ids,
    )
    source_prompt_logits = source_compiled = None
    if require_source_prompt:
        source_prompt_ids = tuple(
            tokenizer.encode(
                _compiler_prompt(row, "source", compiler_input),
                add_special_tokens=False,
            )
        )
        source_prompt_states = _encode_prompt_states(
            model, source_prompt_ids, representation=prompt_representation
        )
        source_prompt_logits, source_compiled = compiled_capsule_action_logits(
            model,
            stitched,
            compiler,
            cached_tokens=cached_tokens,
            prompt_states=source_prompt_states,
            readout_ids=neutral.readout_ids,
            # Teacher-force the common target path up to the first divergence;
            # only the two competing logits at that point are supervised.
            action_ids=action_ids,
        )
    del stitched
    return (
        logits,
        compiled,
        source_prompt_logits,
        source_compiled,
        action_ids,
        source_ids,
    )


def _training_losses(
    logits,
    compiled,
    source_prompt_logits,
    source_compiled,
    action_ids,
    source_ids,
    row,
    compiler,
    *,
    selection_weight,
    selection_window,
    distill_weight,
    contrast_weight,
    contrast_margin,
    conditioning_weight,
    drift_regularization,
):
    device = logits.device
    labels = torch.tensor(action_ids, dtype=torch.long, device=device)
    token_ce = F.cross_entropy(logits.float().squeeze(0), labels, reduction="none")
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
    conditioning = logits.new_zeros((), dtype=torch.float32)
    if difference < min(len(action_ids), len(source_ids)):
        target_logit = logits[0, difference, action_ids[difference]].float()
        source_logit = logits[0, difference, source_ids[difference]].float()
        contrast = F.softplus(contrast_margin - (target_logit - source_logit))
        if source_prompt_logits is not None:
            source_prompt_target = source_prompt_logits[
                0, difference, action_ids[difference]
            ].float()
            source_prompt_source = source_prompt_logits[
                0, difference, source_ids[difference]
            ].float()
            conditioning = F.softplus(
                contrast_margin - (source_prompt_source - source_prompt_target)
            )
    drift = compiler.drift_regularization(compiled)
    if source_compiled is not None:
        drift = 0.5 * (
            drift + compiler.drift_regularization(source_compiled)
        )
    loss = (
        ce
        + distill_weight * distill
        + contrast_weight * contrast
        + conditioning_weight * conditioning
        + drift_regularization * drift
    )
    return {
        "loss": loss,
        "ce": ce,
        "distill": distill,
        "contrast": contrast,
        "conditioning": conditioning,
        "drift": drift,
    }


@torch.inference_mode()
def _evaluate(
    model,
    tokenizer,
    compiler,
    rows,
    *,
    max_new_tokens,
    collect_rows,
    prompt_representation,
    compiler_input="full_prompt",
):
    arms = ("target_prompt", "source_prompt", "initial_capsule", "no_capsule")
    counts = {
        arm: {"target": 0, "source": 0, "valid_json": 0} for arm in arms
    }
    details = []
    for index, row in enumerate(rows, 1):
        _, neutral, stitched, cached_tokens = _student_cache(
            model, tokenizer, row, device=None
        )
        target_prompt_ids = tuple(
            tokenizer.encode(
                _compiler_prompt(row, "target", compiler_input),
                add_special_tokens=False,
            )
        )
        source_prompt_ids = tuple(
            tokenizer.encode(
                _compiler_prompt(row, "source", compiler_input),
                add_special_tokens=False,
            )
        )
        target_prompt_states = _encode_prompt_states(
            model, target_prompt_ids, representation=prompt_representation
        )
        source_prompt_states = _encode_prompt_states(
            model, source_prompt_ids, representation=prompt_representation
        )
        factories = {
            "target_prompt": lambda: start_compiled_capsule_readout(
                model,
                legacy_cache=stitched,
                compiler=compiler,
                cached_tokens=cached_tokens,
                prompt_states=target_prompt_states,
                readout_ids=neutral.readout_ids,
            ),
            "source_prompt": lambda: start_compiled_capsule_readout(
                model,
                legacy_cache=stitched,
                compiler=compiler,
                cached_tokens=cached_tokens,
                prompt_states=source_prompt_states,
                readout_ids=neutral.readout_ids,
            ),
            "initial_capsule": lambda: start_embedding_capsule_readout(
                model,
                legacy_cache=stitched,
                soft_embeddings=compiler.initial_embeddings,
                cached_tokens=cached_tokens,
                readout_ids=neutral.readout_ids,
            ),
            "no_capsule": lambda: start_readout(
                model,
                legacy_cache=stitched,
                cached_tokens=cached_tokens,
                input_ids=neutral.readout_ids,
            ),
        }
        row_detail = {"id": row["id"], "prompt_tokens": len(target_prompt_ids)}
        for arm, factory in factories.items():
            _, text = greedy_action(
                model, factory, tokenizer, max_new_tokens=max_new_tokens
            )
            scored = _score_text(text, row)
            counts[arm]["target"] += int(scored["target"]["compatible_em"])
            counts[arm]["source"] += int(scored["source"]["compatible_em"])
            counts[arm]["valid_json"] += int(scored["call"]["valid_json"])
            if collect_rows:
                row_detail[arm] = scored
        if collect_rows:
            details.append(row_detail)
        if index == 1 or index % 25 == 0:
            print(
                {
                    "evaluation": index,
                    "rows": len(rows),
                    **{
                        arm: f"{counts[arm]['target']}/{index}" for arm in arms
                    },
                },
                flush=True,
            )
        del stitched
        gc.collect()
        torch.cuda.empty_cache()
    count = len(rows)
    metrics = {
        arm: {
            "rows": count,
            "target_correct": value["target"],
            "target_accuracy": value["target"] / count,
            "source_correct": value["source"],
            "source_accuracy": value["source"] / count,
            "valid_json": value["valid_json"],
            "valid_json_rate": value["valid_json"] / count,
        }
        for arm, value in counts.items()
    }
    metrics["conditioning"] = {
        "target_minus_source_prompt_target_accuracy": (
            metrics["target_prompt"]["target_accuracy"]
            - metrics["source_prompt"]["target_accuracy"]
        ),
        "source_prompt_source_minus_target_accuracy": (
            metrics["source_prompt"]["source_accuracy"]
            - metrics["source_prompt"]["target_accuracy"]
        ),
    }
    return metrics, details


def _student_cache(model, tokenizer, row, *, device):
    source = build_chat_segments(
        tokenizer,
        system_prompt=row["source_prompt"],
        tools=row["source_tools"],
        history=row["history"],
    )
    neutral = build_chat_segments(
        tokenizer,
        system_prompt=NEUTRAL_RECEIVER_PROMPT,
        tools=row["target_tools"],
        history=row["history"],
    )
    validate_shared_handoff(source, neutral)
    source_cache = prefill_legacy_cache(
        model, source.prefix_ids + source.history_ids
    )
    neutral_prefix = prefill_legacy_cache(model, neutral.prefix_ids)
    stitched = stitch_history_cache(
        model,
        source_context_cache=source_cache,
        target_prefix_cache=neutral_prefix,
        source_prefix_length=len(source.prefix_ids),
        target_prefix_length=len(neutral.prefix_ids),
        transfer_history_length=len(source.history_ids),
    )
    if device is not None:
        stitched = _cache_to(stitched, device)
    cached_tokens = len(neutral.prefix_ids) + len(neutral.history_ids)
    del source_cache, neutral_prefix
    return source, neutral, stitched, cached_tokens


def _score_text(text: str, row: dict) -> dict:
    call = parse_tool_call(text)
    return {
        "generation": text,
        "call": {
            "name": call.name,
            "arguments": call.arguments,
            "valid_json": call.valid_json,
        },
        "target": score_bfcl_call(
            call,
            expected=row["target_expected"],
            argument_options=row["target_argument_options"],
        ),
        "source": score_bfcl_call(
            call,
            expected=row["source_expected"],
            argument_options=row["source_argument_options"],
        ),
    }


if __name__ == "__main__":
    main()
