from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .data import iter_records


@dataclass(frozen=True)
class MultipleChoiceDocument:
    index: int
    contexts: list[str]
    choices: list[str]
    label: int
    normalization: str
    subject: str | None = None


def iter_arc_challenge(root: str | Path):
    path = Path(root) / "ARC-Challenge" / "test-00000-of-00001.parquet"
    for index, doc in enumerate(iter_records(path)):
        labels = doc["choices"]["label"]
        answer = str(doc["answerKey"])
        if answer not in labels and answer.isdigit():
            answer = str(int(answer))
        yield MultipleChoiceDocument(
            index=index,
            contexts=[f'Question: {doc["question"]}\nAnswer:'],
            choices=list(doc["choices"]["text"]),
            label=labels.index(answer),
            normalization="character",
        )


def iter_winogrande(root: str | Path):
    path = Path(root) / "winogrande_xl" / "validation-00000-of-00001.parquet"
    for index, doc in enumerate(iter_records(path)):
        blank = doc["sentence"].index("_")
        contexts = [
            doc["sentence"][:blank] + doc["option1"],
            doc["sentence"][:blank] + doc["option2"],
        ]
        continuation = doc["sentence"][blank + 1 :].strip()
        yield MultipleChoiceDocument(
            index=index,
            contexts=contexts,
            choices=[continuation, continuation],
            label=int(doc["answer"]) - 1,
            normalization="none",
        )


def iter_mmlu(root: str | Path):
    root = Path(root) / "all"
    dev_by_subject: dict[str, list[dict]] = {}
    for doc in iter_records(root / "dev-00000-of-00001.parquet"):
        dev_by_subject.setdefault(doc["subject"], []).append(doc)
    for subject, examples in dev_by_subject.items():
        if len(examples) < 5:
            raise ValueError(f"MMLU subject {subject!r} has only {len(examples)} dev examples")

    for index, doc in enumerate(iter_records(root / "test-00000-of-00001.parquet")):
        subject = doc["subject"]
        fewshot = [mmlu_question(example) + " " + "ABCD"[int(example["answer"])] for example in dev_by_subject[subject][:5]]
        context = "\n\n".join([*fewshot, mmlu_question(doc)])
        yield MultipleChoiceDocument(
            index=index,
            contexts=[context],
            choices=list("ABCD"),
            label=int(doc["answer"]),
            normalization="none",
            subject=subject,
        )


def mmlu_question(doc: dict) -> str:
    choices = doc["choices"]
    return (
        f'{doc["question"].strip()}\n'
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer:"
    )


GSM8K_FEWSHOT = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
        "5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
)


# Wait for trailing whitespace before accepting a number. Without this guard,
# incremental decoding can mistake the first token of e.g. ``20`` for answer ``2``.
GSM8K_ANSWER_STOP_PATTERN = (
    r"(?i)(?:the\s+)?(?:final\s+)?answer(?:\*{1,2})?\s*"
    r"(?:(?:is|=|should\s+be)\s*:?\s*|:\s*(?:\*{1,2})?\s*)"
    r"(?:\*{1,2})?\$?-?[0-9][0-9,]*(?:\.[0-9]+)?"
    r"(?:\*{1,2})?(?:\.)?(?=\s)"
)


def gsm8k_prompt(question: str) -> str:
    # This is the rendered lm-eval prompt. YAML's folded scalar turns the
    # apparent blank line in gsm8k-cot.yaml into one newline here.
    examples = [f"Q: {q}\nA: {answer}" for q, answer in GSM8K_FEWSHOT]
    return "\n\n".join([*examples, f"Q: {question}\nA:"])


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip()


def gsm8k_extract_strict(response: str) -> str:
    matches = re.findall(r"The answer is (\-?[0-9\.\,]+).", response)
    return matches[0].strip() if matches else "[invalid]"


def gsm8k_extract_flexible(response: str) -> str:
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", response)
    if not matches:
        return "[invalid]"
    return next((item for item in matches[-1] if item), "[invalid]").strip()


def gsm8k_exact_match(prediction: str, gold: str) -> bool:
    def normalize(value: str) -> str:
        value = value.lower().replace(",", "").replace("$", "")
        return re.sub(r"\.$", "", value)

    return normalize(prediction) == normalize(gold)


def wikitext_detokenize(page: str) -> str:
    string = page
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    for before, after in ((" : ", ": "), (" ; ", "; "), (" . ", ". "), (" ! ", "! "), (" ? ", "? "), (" , ", ", ")):
        string = string.replace(before, after)
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r'"\s*([^\"]*?)\s*"', r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    return string.replace(" 's", "'s")


def load_wikitext_tokens(root: str | Path, tokenizer) -> list[int]:
    path = Path(root) / "wikitext-2-raw-v1" / "wikitext-2-raw-v1-test.parquet"
    text = "\n\n".join(wikitext_detokenize(doc["page"]) for doc in iter_records(path))
    return tokenizer(text, add_special_tokens=False)["input_ids"]
