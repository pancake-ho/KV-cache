from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_em(prediction: str, golds: list[str]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(gold) for gold in golds))


def answer_f1(prediction: str, golds: list[str]) -> float:
    return max((_token_f1(prediction, gold) for gold in golds), default=0.0)


def clean_short_answer(text: str) -> str:
    for marker in (
        "<|im_end|>",
        "<|eom_id|>",
        "<|eot_id|>",
        "<|endoftext|>",
    ):
        text = text.replace(marker, "")
    text = text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    answer = lines[0]
    answer = re.sub(r"^(?:final\s+answer|answer)\s*:\s*", "", answer, flags=re.I)
    return answer.strip().strip('"').strip()


def _token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
