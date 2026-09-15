from __future__ import annotations

from typing import Sequence


RESPONSE_TEMPLATE = "The answer is:"


def ensure_question_terminal(question: str) -> str:
    question = question.strip()

    if not question:
        return question

    if question.endswith(("?", ".", ":", "-")):
        return question

    return question + "?"


def format_medmcqa_paper(
    *,
    question: str,
    choices: Sequence[str],
    answer_letter: str,
) -> str:
    """
    Reproduce the prompt structure used in the selected paper's
    public create_dataset.py as closely as possible.
    """
    if len(choices) != 4:
        raise ValueError(
            f"MedMCQA requires 4 choices, got {len(choices)}"
        )

    question = ensure_question_terminal(question)

    rendered = [
        f"MCQ: {question}",
        "Options:",
        f"A. {choices[0]}",
        f"B. {choices[1]}",
        f"C. {choices[2]}",
        f"D. {choices[3]}",
        f"{RESPONSE_TEMPLATE} {answer_letter}",
    ]

    return "\n".join(rendered) + "\n"


def format_pubmedqa_paper(
    *,
    question: str,
    context: str,
    answer: str,
) -> str:
    question = ensure_question_terminal(question)

    rendered = [
        f"Closed Question: {context}",
        question,
        f"{RESPONSE_TEMPLATE} {answer}",
    ]

    return "\n".join(rendered) + "\n"


def format_flashcard_paper(
    *,
    question: str,
    answer: str,
) -> str:
    return (
        f"Open Question: {question}\n"
        f"{RESPONSE_TEMPLATE} {answer}\n"
    )


def format_normalized_qa(
    *,
    question: str,
    answer: str,
    context: str = "",
    choices: Sequence[str] | None = None,
) -> str:
    """
    Dataset-independent representation.

    This is NOT the primary paper-faithful SFT input.
    It is stored now so that a future format-controlled ablation
    can separate dataset/task-format effects from topic effects.
    """
    lines = [
        "Question:",
        question.strip(),
    ]

    if context.strip():
        lines.extend(
            [
                "",
                "Context:",
                context.strip(),
            ]
        )

    if choices:
        lines.extend(
            [
                "",
                "Choices:",
            ]
        )

        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for idx, choice in enumerate(choices):
            if idx >= len(labels):
                raise ValueError(
                    "Too many choices for normalized QA format"
                )

            lines.append(
                f"{labels[idx]}. {choice.strip()}"
            )

    lines.extend(
        [
            "",
            "Answer:",
            answer.strip(),
        ]
    )

    return "\n".join(lines) + "\n"