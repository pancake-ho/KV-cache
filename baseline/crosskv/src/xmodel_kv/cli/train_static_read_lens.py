from __future__ import annotations

import argparse
import gc
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from ..policy_imprinting import (
    ChatSegments,
    action_logits,
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
from .bfcl_authorization_handoff import authorization_cases
from .bfcl_policy_imprinting import _jsonl


@dataclass
class PreparedExample:
    case_index: int
    split: str
    case: object
    target: ChatSegments
    cache_cpu: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    target_action_ids: tuple[int, ...]
    teacher_logits_cpu: torch.Tensor
    source_call: object
    target_call: object
    baseline_call: object


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a static read-time lens for one ALPHA-to-BETA handoff"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--train-cases", type=int, default=12)
    parser.add_argument("--val-cases", type=int, default=6)
    parser.add_argument("--background-records", type=int, default=40)
    parser.add_argument("--min-history-tokens", type=int, default=4096)
    parser.add_argument("--layers", default="18-29")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if min(
        args.train_cases,
        args.val_cases,
        args.background_records,
        args.min_history_tokens,
        args.rank,
        args.epochs,
        args.max_new_tokens,
    ) < 1:
        parser.error("case counts, lengths, rank, epochs, and max-new-tokens must be positive")
    layers = _parse_layers(args.layers)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    register_read_lens_attention()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        attn_implementation="static_read_lens",
    ).eval()
    if max(layers) >= model.config.num_hidden_layers:
        parser.error("layers exceed the model depth")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bank = StaticReadLensBank(model, layers=layers, rank=args.rank).to(
        device=args.device, dtype=torch.float32
    )

    cases = _diverse_cases(
        Path(args.dataset_root),
        train_cases=args.train_cases,
        val_cases=args.val_cases,
        background_records=args.background_records,
        seed=args.seed,
    )
    prepared, preparation_rows = _prepare_examples(
        model,
        tokenizer,
        bank,
        cases,
        min_history_tokens=args.min_history_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    (output_dir / "preparation.json").write_text(
        json.dumps(preparation_rows, ensure_ascii=False, indent=2) + "\n"
    )
    train = [example for example in prepared if example.split == "train"]
    validation = [example for example in prepared if example.split == "validation"]
    if not train or not validation:
        raise SystemExit(
            f"need at least one valid example per split; got train={len(train)}, "
            f"validation={len(validation)}"
        )
    print(
        {
            "prepared_train": len(train),
            "prepared_validation": len(validation),
            "lens_parameters": sum(p.numel() for p in bank.parameters()),
        },
        flush=True,
    )

    optimizer = AdamW(
        bank.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best = None
    for epoch in range(args.epochs + 1):
        if epoch:
            random.Random(args.seed + epoch).shuffle(train)
            bank.train()
            losses = []
            for example in train:
                optimizer.zero_grad(set_to_none=True)
                cache = _cache_to(example.cache_cpu, args.device)
                bank.configure(
                    model,
                    stale_start=len(example.target.prefix_ids),
                    stale_end=len(example.target.prefix_ids)
                    + len(example.target.history_ids),
                    enabled=True,
                )
                logits = _student_action_logits(
                    model,
                    cache,
                    cached_tokens=len(example.target.prefix_ids)
                    + len(example.target.history_ids),
                    readout_ids=example.target.readout_ids,
                    action_ids=example.target_action_ids,
                )
                teacher = example.teacher_logits_cpu.to(
                    device=logits.device, dtype=torch.float32
                )
                student_log_probs = logits.float().log_softmax(dim=-1)
                kl = F.kl_div(
                    student_log_probs,
                    teacher.softmax(dim=-1),
                    reduction="batchmean",
                ) / logits.shape[1]
                labels = torch.tensor(
                    example.target_action_ids, dtype=torch.long, device=logits.device
                )
                ce = F.cross_entropy(logits.float().squeeze(0), labels)
                regularizer = lens_regularization(bank)
                loss = kl + args.ce_weight * ce + args.regularization * regularizer
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bank.parameters(), args.grad_clip)
                optimizer.step()
                losses.append(float(loss.detach()))
                del cache, logits, teacher, student_log_probs, loss
            training_loss = sum(losses) / len(losses)
        else:
            training_loss = None

        bank.eval()
        train_metrics = _evaluate(
            model, tokenizer, bank, train, device=args.device, max_new_tokens=args.max_new_tokens
        )
        val_metrics = _evaluate(
            model,
            tokenizer,
            bank,
            validation,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        row = {
            "epoch": epoch,
            "training_loss": training_loss,
            "train": train_metrics,
            "validation": val_metrics,
            "lens_norms": lens_norms(bank),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = (
            val_metrics["target_accuracy"],
            val_metrics["target_native_agreement"],
            -val_metrics["mean_teacher_kl"],
        )
        if best is None or score > best:
            best = score
            torch.save(
                {
                    "state_dict": bank.state_dict(),
                    "layers": layers,
                    "rank": args.rank,
                    "source_policy": "TENANT-ALPHA",
                    "target_policy": "TENANT-BETA",
                    "epoch": epoch,
                    "validation": val_metrics,
                },
                output_dir / "best.pt",
            )
        (output_dir / "metrics.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n"
        )

    torch.save(
        {
            "state_dict": bank.state_dict(),
            "layers": layers,
            "rank": args.rank,
            "source_policy": "TENANT-ALPHA",
            "target_policy": "TENANT-BETA",
            "epoch": args.epochs,
        },
        output_dir / "last.pt",
    )


def _diverse_cases(root, *, train_cases, val_cases, background_records, seed):
    total = train_cases + val_cases
    base = authorization_cases(root, max_cases=total, seed=seed, background_records=0)
    records = _memory_records(root)
    cutoff = max(background_records, int(len(records) * 0.75))
    train_pool = records[:cutoff]
    validation_pool = records[cutoff:]
    if len(validation_pool) < background_records:
        raise ValueError("not enough held-out BFCL memory records for validation")
    rows = []
    for case_index, (case_map, _) in enumerate(base):
        split = "train" if case_index < train_cases else "validation"
        pool = train_pool if split == "train" else validation_pool
        generator = random.Random(seed * 10_000 + case_index)
        chosen = generator.sample(pool, background_records)
        background = _background_messages(chosen)
        case = case_map["forward"]
        case = replace(case, history=[*background, *case.history])
        rows.append((case_index, split, case))
    return rows


def _memory_records(root):
    records = []
    for domain in ("notetaker", "customer", "healthcare", "student", "finance"):
        path = root / "memory_prereq_conversation" / f"memory_{domain}.json"
        for document in _jsonl(path):
            for turn in document["question"]:
                for message in turn:
                    content = message.get("content")
                    if message.get("role") == "user" and isinstance(content, str):
                        records.append(content)
    return records


def _background_messages(records):
    messages = []
    for content in records:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": (
                        "Completed background record for continuity only; it has no "
                        "pending tool operation:\n" + content
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Background record archived. It does not change the active "
                        "workflow stage."
                    ),
                },
            ]
        )
    return messages


@torch.inference_mode()
def _prepare_examples(
    model,
    tokenizer,
    bank,
    cases,
    *,
    min_history_tokens,
    max_new_tokens,
):
    prepared = []
    rows = []
    bank.disable(model)
    for case_index, split, case in cases:
        source = build_chat_segments(
            tokenizer,
            system_prompt=case.source_prompt,
            tools=case.source_tools,
            history=case.history,
        )
        target = build_chat_segments(
            tokenizer,
            system_prompt=case.target_prompt,
            tools=case.target_tools,
            history=case.history,
        )
        validate_shared_handoff(source, target)
        if len(target.history_ids) < min_history_tokens:
            raise ValueError(
                f"case {case_index} has only {len(target.history_ids)} history tokens"
            )
        source_context = prefill_legacy_cache(
            model, source.prefix_ids + source.history_ids
        )
        target_context = prefill_legacy_cache(
            model, target.prefix_ids + target.history_ids
        )
        source_factory = _native_factory(model, source_context, source)
        target_factory = _native_factory(model, target_context, target)
        source_action_ids, source_text = greedy_action(
            model, source_factory, tokenizer, max_new_tokens=max_new_tokens
        )
        target_action_ids, target_text = greedy_action(
            model, target_factory, tokenizer, max_new_tokens=max_new_tokens
        )
        source_call = parse_tool_call(source_text)
        target_call = parse_tool_call(target_text)
        source_correct = _matches(source_call, case.source_expected)
        target_correct = _matches(target_call, case.target_expected)
        row = {
            "case_index": case_index,
            "split": split,
            "history_tokens": len(target.history_ids),
            "source_correct": source_correct,
            "target_correct": target_correct,
            "source_call": source_call.arguments,
            "target_call": target_call.arguments,
        }
        if source_correct and target_correct and source_call != target_call:
            teacher_logits = action_logits(model, target_factory, target_action_ids)
            target_prefix = prefill_legacy_cache(model, target.prefix_ids)
            stitched = stitch_history_cache(
                model,
                source_context_cache=source_context,
                target_prefix_cache=target_prefix,
                source_prefix_length=len(source.prefix_ids),
                target_prefix_length=len(target.prefix_ids),
                transfer_history_length=len(target.history_ids),
            )
            baseline_factory = _cache_factory(
                model,
                stitched,
                len(target.prefix_ids) + len(target.history_ids),
                target.readout_ids,
            )
            _, baseline_text = greedy_action(
                model, baseline_factory, tokenizer, max_new_tokens=max_new_tokens
            )
            baseline_call = parse_tool_call(baseline_text)
            row["baseline_target_correct"] = _matches(
                baseline_call, case.target_expected
            )
            row["baseline_matches_source"] = baseline_call == source_call
            prepared.append(
                PreparedExample(
                    case_index,
                    split,
                    case,
                    target,
                    _cache_to(stitched, "cpu"),
                    tuple(target_action_ids),
                    teacher_logits.detach().float().cpu(),
                    source_call,
                    target_call,
                    baseline_call,
                )
            )
        rows.append(row)
        print(row, flush=True)
        del source_context, target_context
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return prepared, rows


def _student_action_logits(
    model,
    legacy_cache,
    *,
    cached_tokens,
    readout_ids,
    action_ids,
):
    device = model.model.embed_tokens.weight.device
    input_ids = tuple(readout_ids) + tuple(action_ids[:-1])
    inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
    final_length = cached_tokens + inputs.shape[1]
    cache = DynamicCache(ddp_cache_data=legacy_cache, config=model.config)
    output = model(
        input_ids=inputs,
        attention_mask=torch.ones((1, final_length), dtype=torch.long, device=device),
        position_ids=torch.arange(cached_tokens, final_length, device=device).unsqueeze(0),
        cache_position=torch.arange(cached_tokens, final_length, device=device),
        past_key_values=cache,
        use_cache=False,
        return_dict=True,
    )
    start = len(readout_ids) - 1
    return output.logits[:, start : start + len(action_ids)]


@torch.inference_mode()
def _evaluate(model, tokenizer, bank, examples, *, device, max_new_tokens):
    target_correct = 0
    target_agreement = 0
    source_matches = 0
    kl_values = []
    rows = []
    for example in examples:
        cache = _cache_to(example.cache_cpu, device)
        bank.configure(
            model,
            stale_start=len(example.target.prefix_ids),
            stale_end=len(example.target.prefix_ids) + len(example.target.history_ids),
            enabled=True,
        )
        factory = _cache_factory(
            model,
            cache,
            len(example.target.prefix_ids) + len(example.target.history_ids),
            example.target.readout_ids,
        )
        _, text = greedy_action(
            model, factory, tokenizer, max_new_tokens=max_new_tokens
        )
        call = parse_tool_call(text)
        logits = action_logits(model, factory, example.target_action_ids)
        teacher = example.teacher_logits_cpu.to(logits.device)
        kl = F.kl_div(
            logits.float().log_softmax(dim=-1),
            teacher.softmax(dim=-1),
            reduction="batchmean",
        ) / logits.shape[1]
        correct = _matches(call, example.case.target_expected)
        agrees = call == example.target_call
        matches_source = call == example.source_call
        target_correct += correct
        target_agreement += agrees
        source_matches += matches_source
        kl_values.append(float(kl))
        rows.append(
            {
                "case_index": example.case_index,
                "target_correct": correct,
                "target_native_agreement": agrees,
                "matches_source": matches_source,
                "generation": text,
            }
        )
        del cache, logits, teacher
    count = len(examples)
    return {
        "rows": count,
        "target_accuracy": target_correct / count,
        "target_native_agreement": target_agreement / count,
        "matches_source_native": source_matches / count,
        "mean_teacher_kl": sum(kl_values) / count,
        "examples": rows,
    }


def _native_factory(model, cache, segments):
    return _cache_factory(
        model,
        cache,
        len(segments.prefix_ids) + len(segments.history_ids),
        segments.readout_ids,
    )


def _cache_factory(model, cache, cached_tokens, native_ids):
    return lambda: start_readout(
        model,
        legacy_cache=cache,
        cached_tokens=cached_tokens,
        input_ids=native_ids,
    )


def _cache_to(cache, device):
    return tuple(
        (
            # Preparation runs under inference_mode. Cloning again after the
            # transfer creates ordinary tensors that autograd may safely save
            # while differentiating the lens parameters.
            key.detach().to(device=device).clone(),
            value.detach().to(device=device).clone(),
        )
        for key, value in cache
    )


def _matches(call, expected):
    return (
        call.valid_json
        and call.name == expected["name"]
        and call.arguments == expected["arguments"]
    )


def _parse_layers(value):
    layers = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            if start > end:
                raise ValueError("layer range start exceeds end")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    if not layers or min(layers) < 0:
        raise ValueError("layers must contain non-negative indices")
    return tuple(sorted(layers))


if __name__ == "__main__":
    main()
