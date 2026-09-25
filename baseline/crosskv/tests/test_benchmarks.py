import re

import pyarrow as pa
import pyarrow.parquet as pq

from xmodel_kv.benchmarks import (
    GSM8K_ANSWER_STOP_PATTERN,
    gsm8k_exact_match,
    gsm8k_extract_flexible,
    gsm8k_extract_strict,
    iter_arc_challenge,
    iter_mmlu,
    iter_winogrande,
    mmlu_question,
    wikitext_detokenize,
)


def test_arc_and_winogrande_formatting(tmp_path):
    arc = tmp_path / "arc" / "ARC-Challenge"
    arc.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{"question": "Why?", "choices": {"text": ["x", "y"], "label": ["A", "B"]}, "answerKey": "B"}]
        ),
        arc / "test-00000-of-00001.parquet",
    )
    arc_doc = next(iter_arc_challenge(tmp_path / "arc"))
    assert arc_doc.contexts == ["Question: Why?\nAnswer:"]
    assert arc_doc.choices == ["x", "y"]
    assert arc_doc.label == 1
    assert arc_doc.normalization == "character"

    wino = tmp_path / "wino" / "winogrande_xl"
    wino.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{"sentence": "The _ ran away.", "option1": "cat", "option2": "dog", "answer": "2"}]
        ),
        wino / "validation-00000-of-00001.parquet",
    )
    wino_doc = next(iter_winogrande(tmp_path / "wino"))
    assert wino_doc.contexts == ["The cat", "The dog"]
    assert wino_doc.choices == ["ran away.", "ran away."]
    assert wino_doc.label == 1


def test_mmlu_five_shot_first_n(tmp_path):
    root = tmp_path / "mmlu" / "all"
    root.mkdir(parents=True)
    dev = [
        {"question": f"dev {i}", "subject": "topic", "choices": ["a", "b", "c", "d"], "answer": i % 4}
        for i in range(5)
    ]
    test = [{"question": "test", "subject": "topic", "choices": ["w", "x", "y", "z"], "answer": 2}]
    pq.write_table(pa.Table.from_pylist(dev), root / "dev-00000-of-00001.parquet")
    pq.write_table(pa.Table.from_pylist(test), root / "test-00000-of-00001.parquet")
    doc = next(iter_mmlu(tmp_path / "mmlu"))
    assert doc.contexts[0].count("Answer:") == 6
    assert mmlu_question(dev[0]) + " A" in doc.contexts[0]
    assert doc.contexts[0].endswith(mmlu_question(test[0]))
    assert doc.choices == ["A", "B", "C", "D"]


def test_gsm8k_filters_match_harness_semantics():
    response = "First 12. The answer is 1,234. Then 999"
    assert gsm8k_extract_strict(response) == "1,234."
    assert gsm8k_extract_flexible(response) == "999"
    assert gsm8k_exact_match("$1,234.", "1234")


def test_gsm8k_answer_stop_waits_for_complete_number():
    assert re.search(GSM8K_ANSWER_STOP_PATTERN, "The answer is 20 cups")
    assert re.search(GSM8K_ANSWER_STOP_PATTERN, "**Answer:** 366 downloads")
    assert re.search(GSM8K_ANSWER_STOP_PATTERN, "The answer is 7.5 dollars")
    assert not re.search(GSM8K_ANSWER_STOP_PATTERN, "The answer is 2")
    assert not re.search(GSM8K_ANSWER_STOP_PATTERN, "The answer is 20")


def test_wikitext_detokenizer():
    assert wikitext_detokenize("A @-@ B . \n") == "A-B.\n"
