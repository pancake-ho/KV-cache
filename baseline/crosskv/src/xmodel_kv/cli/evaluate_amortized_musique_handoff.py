from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..coqa import coqa_f1
from ..policy_imprinting import build_chat_segments, prefill_legacy_cache
from ..semantic_kv_summary import concatenate_caches
from ..semantic_kv_transcoder import TailReadoutKVTranscoder
from .probe_semantic_kv_summary import (
    ACK_B,
    AGENT_A_SYSTEM,
    _generate,
    _normalized,
    _split_chat_before_last_user,
)


DEPENDENT_B_SYSTEM = (
    "You are the receiving agent in a two-stage lookup task. The context before the "
    "user request either states or semantically encodes Agent A's intermediate answer. "
    "Find the candidate whose LOOKUP_KEY exactly matches that answer and output its "
    "RESULT. Never output the LOOKUP_KEY, intermediate answer, reasoning, or prose."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained cross-model semantic-KV transcoder on MuSiQue handoff."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--indices", required=True)
    parser.add_argument("--quant-bits", type=int, choices=(4, 8, 16), default=16)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    indices = tuple(int(part) for part in args.indices.split(",") if part.strip())
    if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
        parser.error("indices must be unique non-negative integers")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, args.dtype)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    source_model = _load_model(args.source_model, args.device, dtype)
    target_model = _load_model(args.target_model, args.device, dtype)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint_config = checkpoint["config"]
    transcoder = TailReadoutKVTranscoder(
        source_model.config,
        target_model.config,
        slots=int(checkpoint_config["slots"]),
        latent_dim=int(checkpoint_config["latent_dim"]),
    ).to(args.device)
    transcoder.load_state_dict(checkpoint["state_dict"])
    transcoder.eval()

    cases = _load_cases(args.dataset, indices)
    rows = []
    for index, case in cases:
        row = _evaluate_case(
            source_model,
            source_tokenizer,
            target_model,
            target_tokenizer,
            transcoder,
            index=index,
            case=case,
            quant_bits=None if args.quant_bits == 16 else args.quant_bits,
            max_new_tokens=args.max_new_tokens,
        )
        rows.append(row)
        print(json.dumps({"event": "case", **row}, ensure_ascii=False), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "config": {
            **vars(args),
            "indices": list(indices),
            "slots": transcoder.slots,
            "latent_dim": transcoder.latent_dim,
        },
        "cases": len(rows),
        "student_mean_f1": _mean(row["student_f1"] for row in rows),
        "teacher_mean_f1": _mean(row["teacher_f1"] for row in rows),
        "no_summary_mean_f1": _mean(row["no_summary_f1"] for row in rows),
        "student_accuracy": _mean(row["student_exact"] for row in rows),
        "teacher_accuracy": _mean(row["teacher_exact"] for row in rows),
        "no_summary_accuracy": _mean(row["no_summary_exact"] for row in rows),
        "wins_vs_no_summary": sum(
            row["student_f1"] > row["no_summary_f1"] for row in rows
        ),
        "mean_source_tokens": _mean(row["source_tokens"] for row in rows),
        "wire_payload_bytes": (
            transcoder.slots * transcoder.latent_dim * args.quant_bits / 8
        ),
        "receiver_bf16_cache_bytes": transcoder.receiver_cache_elements * 2,
    }
    with (output_dir / "results.jsonl").open("w") as result_file:
        for row in rows:
            result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)


def _load_model(path: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def _evaluate_case(
    source_model,
    source_tokenizer,
    target_model,
    target_tokenizer,
    transcoder,
    *,
    index: int,
    case: dict,
    quant_bits: int | None,
    max_new_tokens: int,
) -> dict:
    source_user = _format_source(case)
    source_segments = build_chat_segments(
        source_tokenizer,
        system_prompt=AGENT_A_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": source_user}],
        enable_thinking=False,
    )
    source_ids = [
        *source_segments.prefix_ids,
        *source_segments.history_ids,
        *source_segments.readout_ids,
    ]
    source_cache = prefill_legacy_cache(source_model, source_ids)

    target_task = _format_target(case)
    student_segments = build_chat_segments(
        target_tokenizer,
        system_prompt=DEPENDENT_B_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": target_task}],
        enable_thinking=False,
    )
    target_system_ids = list(student_segments.prefix_ids)
    target_system_cache = prefill_legacy_cache(target_model, target_system_ids)
    student_query_ids = [
        *student_segments.history_ids,
        *student_segments.readout_ids,
    ]

    # The dense teacher is a controlled handoff oracle: it receives the exact
    # Stage-A intermediate answer in text and therefore isolates Stage-B lookup
    # quality.  Asking it to rescan the 6k-token source dossier would conflate
    # source reasoning failures with semantic-state transfer failures.
    teacher_history = [
        {
            "role": "user",
            "content": (
                "Agent A intermediate answer: " + case["agent_a"]["gold_answer"]
            ),
        },
        {"role": "assistant", "content": ACK_B},
    ]
    teacher_prefix_ids, teacher_query_ids, teacher_system_ids = (
        _split_chat_before_last_user(
            target_tokenizer,
            system_prompt=DEPENDENT_B_SYSTEM,
            prior_history=teacher_history,
            query=target_task,
        )
    )
    if teacher_system_ids != target_system_ids:
        raise ValueError("teacher and student target system prefixes differ")
    teacher_cache = prefill_legacy_cache(target_model, teacher_prefix_ids)

    latent, compact_cache = transcoder(
        source_model,
        target_model,
        source_cache=source_cache,
        source_cached_tokens=len(source_ids),
        target_start=len(target_system_ids),
        quant_bits=quant_bits,
    )
    student_cache = concatenate_caches(target_system_cache, compact_cache)
    student_text = _generate(
        target_model,
        target_tokenizer,
        legacy_cache=student_cache,
        cached_tokens=len(target_system_ids) + transcoder.slots,
        query_ids=student_query_ids,
        max_new_tokens=max_new_tokens,
    )
    teacher_text = _generate(
        target_model,
        target_tokenizer,
        legacy_cache=teacher_cache,
        cached_tokens=len(teacher_prefix_ids),
        query_ids=teacher_query_ids,
        max_new_tokens=max_new_tokens,
    )
    no_summary_text = _generate(
        target_model,
        target_tokenizer,
        legacy_cache=target_system_cache,
        cached_tokens=len(target_system_ids),
        query_ids=student_query_ids,
        max_new_tokens=max_new_tokens,
    )
    gold = case["agent_b"]["gold_answer"]
    student_answer = _clean_generation(student_text)
    teacher_answer = _clean_generation(teacher_text)
    no_summary_answer = _clean_generation(no_summary_text)
    return {
        "index": index,
        "id": case["id"],
        "gold": gold,
        "source_tokens": len(source_ids),
        "target_teacher_tokens": len(teacher_prefix_ids),
        "latent_abs_mean": float(latent.float().abs().mean()),
        "student": student_answer,
        "teacher": teacher_answer,
        "no_summary": no_summary_answer,
        "student_f1": coqa_f1(student_answer, [gold]),
        "teacher_f1": coqa_f1(teacher_answer, [gold]),
        "no_summary_f1": coqa_f1(no_summary_answer, [gold]),
        "student_exact": float(_normalized(student_answer) == _normalized(gold)),
        "teacher_exact": float(_normalized(teacher_answer) == _normalized(gold)),
        "no_summary_exact": float(_normalized(no_summary_answer) == _normalized(gold)),
    }


def _format_source(case: dict) -> str:
    documents = "\n\n".join(
        f"[DOC-{position}] {document['title']}\n{document['text']}"
        for position, document in enumerate(case["agent_a"]["documents"], 1)
    )
    return (
        f"<STAGE_A_DOSSIER>\n{documents}\n</STAGE_A_DOSSIER>\n\n"
        f"Stage-A question: {case['agent_a']['question']}\n"
        "Internally retain the short intermediate answer for the receiving agent."
    )


def _format_target(case: dict) -> str:
    candidates = "\n\n".join(
        f"[{candidate['candidate_id']}]\n"
        f"LOOKUP_KEY: {candidate['bridge_answer']}\n"
        f"RESULT: {candidate['final_answer']}"
        for candidate in case["agent_b"]["candidates"]
    )
    return (
        "Use the handed-off state as Agent A's intermediate answer.\n\n"
        f"<STAGE_B_CANDIDATES>\n{candidates}\n</STAGE_B_CANDIDATES>\n\n"
        f"Stage-B question: {case['agent_b']['question_template']}\n"
        "#1 means the answer retained from Stage A. Output only the short RESULT."
    )


def _clean_generation(text: str) -> str:
    for stop_text in ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>"):
        text = text.split(stop_text, 1)[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def _load_cases(path: str, indices: tuple[int, ...]) -> list[tuple[int, dict]]:
    wanted = set(indices)
    selected = []
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            if index in wanted:
                selected.append((index, json.loads(line)))
    if {index for index, _ in selected} != wanted:
        raise ValueError("dataset does not contain every requested index")
    return sorted(selected)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
