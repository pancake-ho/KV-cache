from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from xmodel_kv.hotpotqa import answer_em, answer_f1, clean_short_answer
from xmodel_kv.policy_imprinting import build_chat_segments, start_readout

from .evaluate_hotpot_summary_transfer import (
    QUESTION_B_SYSTEM,
    generate_from_state,
    load_slice,
    prepare_target_spec,
    timed_cuda,
)
from .train_musique_semantic_handoff import _load_model


ARMS = (
    "plaintext_source",
    "plaintext_shifted",
    "plaintext_gold",
    "full_recompute",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate plaintext-answer handoff and full target-side recompute on "
            "the same Hotpot rows used by the compact-KV transfer experiment."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-results", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    if args.offset < 0 or min(args.count, args.max_new_tokens) < 1:
        parser.error("offset must be non-negative and counts positive")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    cases = load_slice(args.dataset, offset=args.offset, count=args.count)
    source_rows = load_source_rows(args.source_results)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, args.device, getattr(torch, args.dtype))
    target_spec = prepare_target_spec(model, tokenizer, "question_conditioned")

    rows = []
    for position, (index, document) in enumerate(cases, 1):
        case_id = document.get("_id", str(index))
        if case_id not in source_rows:
            raise ValueError(f"source results do not contain case {case_id}")
        source = source_rows[case_id]
        shifted_id = source["shifted_capsule_source_id"]
        if shifted_id not in source_rows:
            raise ValueError(f"source results do not contain shifted case {shifted_id}")
        if source["question"] != document["input"]:
            raise ValueError(f"dataset/source question mismatch for {case_id}")
        summaries = {
            "plaintext_source": source["source_answer"],
            "plaintext_shifted": source_rows[shifted_id]["source_answer"],
            "plaintext_gold": document["answers"][0],
        }
        row = {
            "index": index,
            "id": case_id,
            "dataset": document.get("dataset", "hotpotqa"),
            "question": document["input"],
            "gold_answers": list(document["answers"]),
            "source_answer": source["source_answer"],
            "source_answer_em": source["source_answer_em"],
            "source_answer_f1": source["source_answer_f1"],
            "shifted_source_id": shifted_id,
        }
        for arm in ARMS:
            summary = summaries.get(arm)
            result = evaluate_arm(
                model,
                tokenizer,
                target_spec=target_spec,
                document=document,
                summary=summary,
                max_new_tokens=args.max_new_tokens,
            )
            generation = result.pop("generation")
            row[arm] = generation
            row[f"{arm}_em"] = answer_em(generation, document["answers"])
            row[f"{arm}_f1"] = answer_f1(generation, document["answers"])
            for key, value in result.items():
                row[f"{arm}_{key}"] = value
            if summary is not None:
                row[f"{arm}_handoff_utf8_bytes"] = len(summary.encode("utf-8"))
        rows.append(row)
        print(json.dumps({"event": "evaluate", **row}, ensure_ascii=False), flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {"event": "progress", "position": position, "count": len(cases)}
            ),
            flush=True,
        )

    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    summary = summarize(rows, config=config)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)


def load_source_rows(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    if path.is_dir():
        path = path / "results.jsonl"
    selected: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("protocol") != "question_conditioned":
                continue
            required = {
                "id",
                "question",
                "source_answer",
                "source_answer_em",
                "source_answer_f1",
                "shifted_capsule_source_id",
            }
            if not required.issubset(row):
                raise ValueError("question-conditioned source row is incomplete")
            if row["id"] in selected:
                raise ValueError(f"duplicate question-conditioned row: {row['id']}")
            selected[row["id"]] = row
    if not selected:
        raise ValueError("source results contain no question-conditioned rows")
    return selected


def format_plaintext_user(summary: str, question: str) -> str:
    return (
        "Agent A handed off the following plaintext summary:\n"
        f"<SUMMARY>{summary.strip()}</SUMMARY>\n\n"
        f"Question: {question.strip()}\n"
        "Return only the exact short answer without explanation."
    )


def format_full_recompute_user(document: dict) -> str:
    return (
        f"<DOSSIER>\n{document['context'].strip()}\n</DOSSIER>\n\n"
        f"Question: {document['input'].strip()}\n"
        "Return only the exact short answer without explanation."
    )


@torch.inference_mode()
def evaluate_arm(
    model,
    tokenizer,
    *,
    target_spec: dict,
    document: dict,
    summary: str | None,
    max_new_tokens: int,
) -> dict:
    user = (
        format_full_recompute_user(document)
        if summary is None
        else format_plaintext_user(summary, document["input"])
    )
    segments = build_chat_segments(
        tokenizer,
        system_prompt=QUESTION_B_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": user}],
        enable_thinking=False,
    )
    if tuple(segments.prefix_ids) != tuple(target_spec["system_ids"]):
        raise ValueError("target system prefix changed across baseline arms")
    query_ids = (*segments.history_ids, *segments.readout_ids)
    state, prefill_ms = timed_cuda(
        lambda: start_readout(
            model,
            legacy_cache=target_spec["system_cache"],
            cached_tokens=len(target_spec["system_ids"]),
            input_ids=query_ids,
        )
    )
    generated_ids, generated_state, generation_ms = generate_from_state(
        model,
        tokenizer,
        state,
        max_new_tokens=max_new_tokens,
    )
    generation = clean_short_answer(
        tokenizer.decode(generated_ids, skip_special_tokens=False)
    )
    result = {
        "generation": generation,
        "prompt_tokens": len(query_ids),
        "prefill_ms": prefill_ms,
        "generation_ms": generation_ms,
    }
    if summary is not None:
        result["handoff_tokens"] = len(
            tokenizer.encode(summary, add_special_tokens=False)
        )
    del state, generated_state
    return result


def summarize(rows: list[dict], *, config: dict | None = None) -> dict:
    if not rows:
        raise ValueError("cannot summarize empty baseline rows")
    output = {"config": config or {}, "cases": len(rows), "arms": {}}
    for arm in ARMS:
        mean = lambda suffix: sum(float(row[f"{arm}_{suffix}"]) for row in rows) / len(rows)
        arm_summary = {
            "em": mean("em"),
            "f1": mean("f1"),
            "mean_prompt_tokens": mean("prompt_tokens"),
            "mean_prefill_ms": mean("prefill_ms"),
            "mean_generation_ms": mean("generation_ms"),
        }
        if arm != "full_recompute":
            arm_summary["mean_handoff_tokens"] = mean("handoff_tokens")
            arm_summary["mean_handoff_utf8_bytes"] = mean("handoff_utf8_bytes")
        output["arms"][arm] = arm_summary
    return output


if __name__ == "__main__":
    main()
