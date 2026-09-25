from __future__ import annotations

import argparse
import gc
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..policy_imprinting import build_chat_segments
from .policy_imprinting import ControlledCase, _run_case


FAMILIES = ("position", "content")
ARMS = ("null", "matched", "third")
LAYOUTS = ("clean", "shifted")
POSITION_DIRECTIONS = ("target_first", "target_last")
CONTENT_DIRECTIONS = ("target_urgent", "target_cheapest")
RECORD_NAMES = (
    "ALPHA",
    "BRAVO",
    "CEDAR",
    "DELTA",
    "EMBER",
    "FALCON",
    "GAMMA",
    "HARBOR",
    "IVORY",
    "JASPER",
    "KAPPA",
    "LOTUS",
    "MAPLE",
    "NOVA",
    "ONYX",
    "PANDA",
    "QUARTZ",
    "RAVEN",
    "SOLAR",
    "TANGO",
    "UMBRA",
    "VIOLET",
    "WILLOW",
    "XENON",
    "YARROW",
    "ZEPHYR",
    "AURORA",
    "BIRCH",
    "CORAL",
    "DRIFT",
    "EAGLE",
    "FROST",
    "GLADE",
    "HELIOS",
    "INDIGO",
    "JUNIPER",
    "KESTREL",
    "LUNAR",
    "METEOR",
    "NEBULA",
)


@dataclass(frozen=True)
class TriadGroup:
    family: str
    direction: str
    replicate: int
    seed: int
    record_ids: tuple[str, ...]
    target_record_id: str
    third_record_id: str
    target_policy: str
    third_policy: str
    cases: dict[str, ControlledCase]
    metadata: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Three-arm causal policy control with >=8 actions and clean/shifted prefixes"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--layouts", default=",".join(LAYOUTS))
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-count", type=int, default=9)
    parser.add_argument("--replay-tokens", default="0,100000")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--shift-padding-tokens", type=int, default=32)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    families = _csv(args.families, FAMILIES, "families")
    arms = _csv(args.arms, ARMS, "arms")
    layouts = _csv(args.layouts, LAYOUTS, "layouts")
    try:
        replay_tokens = tuple(int(part) for part in args.replay_tokens.split(","))
    except ValueError as error:
        raise SystemExit("replay-tokens must be comma-separated integers") from error
    if any(value < 0 for value in replay_tokens):
        parser.error("replay-tokens must be non-negative")
    if args.record_count < 8:
        parser.error("record-count must be at least 8")
    if args.record_count > len(RECORD_NAMES):
        parser.error(f"record-count cannot exceed {len(RECORD_NAMES)}")
    if args.replicates < 1 or args.max_new_tokens < 1:
        parser.error("replicates and max-new-tokens must be positive")
    if args.shift_padding_tokens < 1:
        parser.error("shift-padding-tokens must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num-shards >= 1 and 0 <= shard-index < num-shards")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    groups = [
        _triad_group(
            family,
            direction,
            replicate=replicate,
            seed=args.seed,
            record_count=args.record_count,
        )
        for family in families
        for direction in _directions(family)
        for replicate in range(args.replicates)
    ]
    prepared = []
    for group_index, group in enumerate(groups):
        if group_index % args.num_shards != args.shard_index:
            continue
        clean_cases, clean_prefix_length = _length_match_group(tokenizer, group)
        for arm in arms:
            for layout in layouts:
                case = clean_cases[arm]
                if layout == "shifted":
                    case = replace(
                        case,
                        source_prompt=_shift_prompt(
                            case.source_prompt, args.shift_padding_tokens
                        ),
                    )
                triad_id = (
                    f"triad.{group.family}.{group.direction}.r{group.replicate}."
                    f"{arm}.{layout}"
                )
                prepared.append(
                    (triad_id, group, arm, layout, clean_prefix_length, case)
                )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed(output_path)
    pending = [item for item in prepared if item[0] not in completed]
    if not pending:
        print({"processed_this_run": 0, "message": "selected triad shard is complete"})
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()
    processed = 0
    with output_path.open("a") as output:
        for triad_id, group, arm, layout, clean_prefix_length, case in pending:
            row = _run_case(
                model,
                tokenizer,
                case,
                replay_tokens=replay_tokens,
                max_new_tokens=args.max_new_tokens,
            )
            zero = row["hybrid"][0]
            source_call = row["source_native"]["tool_call"]
            target_call = row["target_native"]["tool_call"]
            candidate_call = zero["tool_call"]
            source_valid = _valid_record_call(source_call, group.record_ids)
            target_valid = _valid_record_call(target_call, group.record_ids)
            candidate_valid = _valid_record_call(candidate_call, group.record_ids)
            source_policy_correct = (
                source_valid
                if arm == "null"
                else source_call == case.source_expected | {"valid_json": True}
            )
            row.update(
                {
                    "triad_id": triad_id,
                    "triad": {
                        "family": group.family,
                        "direction": group.direction,
                        "arm": arm,
                        "layout": layout,
                        "replicate": group.replicate,
                        "seed": group.seed,
                        "record_count": len(group.record_ids),
                        "chance_level": 1.0 / len(group.record_ids),
                        "record_ids": group.record_ids,
                        "target_record_id": group.target_record_id,
                        "third_record_id": group.third_record_id,
                        "target_policy": group.target_policy,
                        "source_policy": (
                            "neutral" if arm == "null" else (
                                group.target_policy if arm == "matched" else group.third_policy
                            )
                        ),
                        "clean_prefix_length": clean_prefix_length,
                        "actual_prefix_shift": (
                            row["token_lengths"]["source_prefix"]
                            - row["token_lengths"]["target_prefix"]
                        ),
                        **group.metadata,
                    },
                    "triad_validity": {
                        "target_native_valid_action": target_valid,
                        "target_policy_correct": (
                            target_call == case.target_expected | {"valid_json": True}
                        ),
                        "source_native_valid_action": source_valid,
                        "source_policy_correct": source_policy_correct,
                        "hybrid_valid_action": candidate_valid,
                    },
                    "triad_outcome": {
                        "hybrid_matches_target_native": candidate_call == target_call,
                        "hybrid_matches_source_native": candidate_call == source_call,
                        "hybrid_matches_target_policy": (
                            candidate_call == case.target_expected | {"valid_json": True}
                        ),
                        "hybrid_matches_third_policy": (
                            candidate_call["name"] == "choose_record"
                            and candidate_call["arguments"]
                            == {"record_id": group.third_record_id}
                            and candidate_call["valid_json"]
                        ),
                    },
                }
            )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            processed += 1
            print(
                {
                    "triad_id": triad_id,
                    "prefix_shift": row["triad"]["actual_prefix_shift"],
                    "A": target_call.get("arguments"),
                    "S": source_call.get("arguments"),
                    "B": candidate_call.get("arguments"),
                    "A_correct": row["triad_validity"]["target_policy_correct"],
                    "S_correct": source_policy_correct,
                    "B=A": row["triad_outcome"]["hybrid_matches_target_native"],
                    "B=S": row["triad_outcome"]["hybrid_matches_source_native"],
                    "B=third": row["triad_outcome"]["hybrid_matches_third_policy"],
                },
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(
        {
            "processed_this_run": processed,
            "shard": f"{args.shard_index}/{args.num_shards}",
            "output": str(output_path),
        }
    )


def _triad_group(
    family: str,
    direction: str,
    *,
    replicate: int,
    seed: int,
    record_count: int,
) -> TriadGroup:
    rng = random.Random(seed + 104_729 * replicate + (0 if family == "position" else 9_973))
    names = rng.sample(RECORD_NAMES, record_count)
    rng.shuffle(names)
    record_ids = tuple(f"REC-{name}" for name in names)
    tool = _choose_record_tool(record_ids)
    if family == "position":
        return _position_group(
            direction,
            replicate=replicate,
            seed=seed,
            record_ids=record_ids,
            tool=tool,
        )
    if family == "content":
        return _content_group(
            direction,
            replicate=replicate,
            seed=seed,
            record_ids=record_ids,
            tool=tool,
            rng=rng,
        )
    raise ValueError(family)


def _position_group(direction, *, replicate, seed, record_ids, tool) -> TriadGroup:
    if direction not in POSITION_DIRECTIONS:
        raise ValueError(direction)
    target_index = 0 if direction == "target_first" else len(record_ids) - 1
    middle_index = len(record_ids) // 2
    target_word = "FIRST" if target_index == 0 else "LAST"
    target_synonym = "INITIAL" if target_index == 0 else "FINAL"
    history: list[dict[str, str]] = []
    for index, record_id in enumerate(record_ids):
        history.extend(
            [
                {
                    "role": "user",
                    "content": (
                        f"Record {index + 1} of {len(record_ids)}. Record ID: {record_id}. "
                        "Category: standard. Status: eligible."
                    ),
                },
                {"role": "assistant", "content": "Record acknowledged."},
            ]
        )
    history.append(
        {"role": "user", "content": "The batch is complete. Select one record now."}
    )
    target_prompt = (
        f"Selection policy: {target_word}. Select the record ID from the {target_word} "
        "position in the batch. Ignore all content fields. Call choose_record exactly once "
        "and return no prose."
    )
    matched_prompt = (
        f"Routing rule: {target_synonym}. Pick the ID appearing in the {target_synonym} "
        "batch position. Record contents are irrelevant. Use choose_record exactly once "
        "with no prose."
    )
    third_prompt = (
        f"Selection policy: MIDDLE. Select the record ID from the MIDDLE position, which is "
        f"record {(len(record_ids) // 2) + 1} of {len(record_ids)}. Ignore all content "
        "fields. Call choose_record exactly once and return no prose."
    )
    null_prompt = (
        "No positional selection policy is active. At batch completion, choose any one "
        "eligible record using your default preference. Call choose_record exactly once "
        "and return no prose."
    )
    return _make_group(
        family="position",
        direction=direction,
        replicate=replicate,
        seed=seed,
        record_ids=record_ids,
        target_record_id=record_ids[target_index],
        third_record_id=record_ids[middle_index],
        target_policy=target_word.lower(),
        third_policy="middle",
        target_prompt=target_prompt,
        matched_prompt=matched_prompt,
        third_prompt=third_prompt,
        null_prompt=null_prompt,
        history=history,
        tool=tool,
        metadata={
            "target_position": target_index,
            "third_position": middle_index,
        },
    )


def _content_group(direction, *, replicate, seed, record_ids, tool, rng) -> TriadGroup:
    if direction not in CONTENT_DIRECTIONS:
        raise ValueError(direction)
    urgent_index, cheapest_index = rng.sample(range(len(record_ids)), 2)
    prices = rng.sample(range(40, 180), len(record_ids))
    prices[cheapest_index] = 7
    history: list[dict[str, str]] = []
    for index, record_id in enumerate(record_ids):
        priority = "URGENT" if index == urgent_index else "ROUTINE"
        history.extend(
            [
                {
                    "role": "user",
                    "content": (
                        f"Record ID: {record_id}. Priority: {priority}. "
                        f"Price: ${prices[index]}. Status: eligible."
                    ),
                },
                {"role": "assistant", "content": "Record acknowledged."},
            ]
        )
    history.append(
        {"role": "user", "content": "The batch is complete. Select one record now."}
    )
    prompts = {
        "urgent": (
            "Content policy: URGENT. Select the unique record whose Priority field is "
            "URGENT, regardless of price or position. Call choose_record exactly once and "
            "return no prose."
        ),
        "urgent_matched": (
            "Routing criterion: HIGH PRIORITY. Pick the sole batch entry explicitly marked "
            "URGENT. Ignore its price and location. Use choose_record exactly once with no prose."
        ),
        "cheapest": (
            "Content policy: CHEAPEST. Select the unique record with the lowest numeric "
            "Price, regardless of priority or position. Call choose_record exactly once and "
            "return no prose."
        ),
        "cheapest_matched": (
            "Routing criterion: MINIMUM PRICE. Pick the sole batch entry whose numeric price "
            "is lower than every other entry. Ignore priority and location. Use choose_record "
            "exactly once with no prose."
        ),
    }
    null_prompt = (
        "No content selection policy is active. At batch completion, choose any one eligible "
        "record using your default preference. Call choose_record exactly once and return no prose."
    )
    if direction == "target_urgent":
        target_policy, third_policy = "urgent", "cheapest"
        target_index, third_index = urgent_index, cheapest_index
    else:
        target_policy, third_policy = "cheapest", "urgent"
        target_index, third_index = cheapest_index, urgent_index
    return _make_group(
        family="content",
        direction=direction,
        replicate=replicate,
        seed=seed,
        record_ids=record_ids,
        target_record_id=record_ids[target_index],
        third_record_id=record_ids[third_index],
        target_policy=target_policy,
        third_policy=third_policy,
        target_prompt=prompts[target_policy],
        matched_prompt=prompts[f"{target_policy}_matched"],
        third_prompt=prompts[third_policy],
        null_prompt=null_prompt,
        history=history,
        tool=tool,
        metadata={
            "urgent_position": urgent_index,
            "cheapest_position": cheapest_index,
            "prices": prices,
        },
    )


def _make_group(
    *,
    family,
    direction,
    replicate,
    seed,
    record_ids,
    target_record_id,
    third_record_id,
    target_policy,
    third_policy,
    target_prompt,
    matched_prompt,
    third_prompt,
    null_prompt,
    history,
    tool,
    metadata,
) -> TriadGroup:
    target_expected = _expected(target_record_id)
    cases = {}
    for arm, source_prompt in {
        "null": null_prompt,
        "matched": matched_prompt,
        "third": third_prompt,
    }.items():
        source_expected = (
            _expected("__UNSPECIFIED__")
            if arm == "null"
            else target_expected if arm == "matched" else _expected(third_record_id)
        )
        cases[arm] = ControlledCase(
            scenario=f"triad_{family}_{arm}",
            direction=direction,
            history_turn_pairs=len(record_ids),
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            source_tools=[tool],
            target_tools=[tool],
            history=history,
            source_expected=source_expected,
            target_expected=target_expected,
            replicate=replicate,
            seed=seed,
        )
    return TriadGroup(
        family=family,
        direction=direction,
        replicate=replicate,
        seed=seed,
        record_ids=record_ids,
        target_record_id=target_record_id,
        third_record_id=third_record_id,
        target_policy=target_policy,
        third_policy=third_policy,
        cases=cases,
        metadata=metadata,
    )


def _length_match_group(tokenizer, group: TriadGroup):
    reference = group.cases["null"]
    prompts = {"target": reference.target_prompt}
    prompts.update({arm: case.source_prompt for arm, case in group.cases.items()})
    candidates = {
        name: _padding_candidates(
            tokenizer,
            prompt,
            tools=reference.target_tools,
            history=reference.history,
        )
        for name, prompt in prompts.items()
    }
    common = set.intersection(*(set(items) for items in candidates.values()))
    if not common:
        raise ValueError(
            f"could not token-length-match prompts for {group.family}/{group.direction}"
        )
    common_length = min(common)
    target_prompt = candidates["target"][common_length]
    cases = {
        arm: replace(
            case,
            source_prompt=candidates[arm][common_length],
            target_prompt=target_prompt,
        )
        for arm, case in group.cases.items()
    }
    for case in cases.values():
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
        if len(source.prefix_ids) != len(target.prefix_ids):
            raise AssertionError("clean triad prefix lengths are not equal")
    return cases, common_length


def _padding_candidates(tokenizer, prompt, *, tools, history, max_padding=64):
    lengths = {}
    for count in range(max_padding + 1):
        padded = (
            prompt
            + "\nInert labels with no policy meaning; ignore them:"
            + " PAD" * count
            + "."
        )
        length = _private_prefix_length(
            tokenizer, padded, tools=tools, history=history
        )
        lengths.setdefault(length, padded)
    return lengths


def _private_prefix_length(tokenizer, prompt, *, tools, history) -> int:
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": prompt}],
            tools=tools,
            add_generation_prompt=False,
            enable_thinking=False,
            tokenize=True,
        )
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
        return len(ids)
    except Exception:
        return len(
            build_chat_segments(
                tokenizer, system_prompt=prompt, tools=tools, history=history
            ).prefix_ids
        )


def _shift_prompt(prompt: str, padding_tokens: int) -> str:
    return (
        prompt
        + "\nAdditional inert source-only labels; ignore them:"
        + " SHIFT_PAD" * padding_tokens
        + "."
    )


def _choose_record_tool(record_ids) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "choose_record",
            "description": "Choose exactly one eligible record from the batch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "enum": list(record_ids)}
                },
                "required": ["record_id"],
            },
        },
    }


def _expected(record_id: str) -> dict[str, Any]:
    return {"name": "choose_record", "arguments": {"record_id": record_id}}


def _valid_record_call(call, record_ids) -> bool:
    return (
        call.get("valid_json") is True
        and call.get("name") == "choose_record"
        and isinstance(call.get("arguments"), dict)
        and call["arguments"].get("record_id") in record_ids
    )


def _directions(family: str):
    return POSITION_DIRECTIONS if family == "position" else CONTENT_DIRECTIONS


def _csv(value: str, allowed, name: str):
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(result) - set(allowed)
    if not result or unknown:
        raise SystemExit(f"invalid {name}: {sorted(unknown)}")
    return result


def _completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {
            row["triad_id"]
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
            if "triad_id" in row
        }


if __name__ == "__main__":
    main()
