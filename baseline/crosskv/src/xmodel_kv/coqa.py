from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable


COQA_DOMAINS = ("mctest", "race", "cnn", "wikipedia", "gutenberg")
COQA_PAPER_TURNS = (1, 3, 5, 7, 10)


def questions(document: dict) -> list[str]:
    value = document["questions"]
    if isinstance(value, dict):
        value = value["input_text"]
    return list(value)


def primary_answers(document: dict) -> list[str]:
    return list(document["answers"]["input_text"])


def context_at_turn(document: dict, turn: int) -> str:
    """Build the official lm-eval CoQA completion prompt at a 1-based turn."""
    all_questions = questions(document)
    all_answers = primary_answers(document)
    if not 1 <= turn <= len(all_questions):
        raise ValueError(f"turn {turn} is outside conversation length {len(all_questions)}")

    parts = [document["story"], "\n\n"]
    for index in range(turn - 1):
        parts.append(f"Q: {all_questions[index]}\n\nA: {all_answers[index]}\n\n")
    parts.append(f"Q: {all_questions[turn - 1]}\n\nA:")
    return "".join(parts)


def gold_answers_at_turn(document: dict, turn: int) -> list[str]:
    if not 1 <= turn <= len(primary_answers(document)):
        raise ValueError(f"turn {turn} is outside the answer sequence")
    candidates = [primary_answers(document)[turn - 1]]
    additional = document.get("additional_answers") or {}
    for answer_set in additional.values():
        candidates.append(answer_set["input_text"][turn - 1])

    unique: list[str] = []
    seen = set()
    for answer in candidates:
        lowered = answer.lower()
        if lowered not in seen:
            unique.append(answer)
            seen.add(lowered)
    return unique


def answer_from_generation(generation: str) -> str:
    return generation.strip().split("\n", 1)[0]


def token_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = _normalize(prediction).split()
    ground_truth_tokens = _normalize(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(ground_truth_tokens)
    return 2.0 * precision * recall / (precision + recall)


def coqa_f1(prediction: str, ground_truths: list[str]) -> float:
    """Match lm-eval/official CoQA leave-one-annotator-out F1 aggregation."""
    if not ground_truths:
        raise ValueError("CoQA requires at least one ground-truth answer")
    if len(ground_truths) == 1:
        return token_f1(prediction, ground_truths[0])

    total = 0.0
    for held_out in range(len(ground_truths)):
        references = ground_truths[:held_out] + ground_truths[held_out + 1 :]
        total += max(token_f1(prediction, answer) for answer in references)
    return total / len(ground_truths)


def select_conversations(
    indexed_documents: Iterable[tuple[int, dict]],
    *,
    per_domain: int,
    strategy: str = "first",
    seed: int = 0,
    domains: tuple[str, ...] = COQA_DOMAINS,
) -> list[tuple[int, dict]]:
    """Choose a balanced subset while retaining original validation indices."""
    import random

    grouped = {domain: [] for domain in domains}
    for index, document in indexed_documents:
        domain = document.get("source")
        if domain in grouped:
            grouped[domain].append((index, document))

    selected = []
    generator = random.Random(seed)
    for domain in domains:
        candidates = grouped[domain]
        if len(candidates) < per_domain:
            raise ValueError(
                f"domain {domain!r} only has {len(candidates)} conversations; need {per_domain}"
            )
        if strategy == "first":
            choice = candidates[:per_domain]
        elif strategy == "random":
            choice = generator.sample(candidates, per_domain)
        else:
            raise ValueError(f"unknown selection strategy: {strategy}")
        selected.extend(choice)
    return selected


def _normalize(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        return "".join(character for character in value if character not in string.punctuation)

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())
