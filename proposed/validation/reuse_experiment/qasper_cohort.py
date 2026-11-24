"""Build a deterministic multi-question cohort from the QASPER test archive."""

import hashlib
import json
import re
import tarfile
from random import Random

from ...src.exp.mapper_optim.inputs import token_hash
from ..e2e_inputs import check_tokenizers, longbench_rows, render_qa_prompt, sha


def words(text):
    return re.findall(r"\w+", text.casefold())


def shingles(terms):
    width = 16
    return zip(*(terms[offset:] for offset in range(width)))


def calibration_shingles(reference):
    items = reference["calibration_documents"]
    if not items:
        raise ValueError("The mapper reference must include its calibration documents")
    rows = longbench_rows(reference["dataset"], {item["dataset"] for item in items})
    result = set()
    for item in items:
        context = rows[item["dataset"]][item["line"]]["context"]
        if hashlib.sha256(context.strip().encode()).hexdigest() != item["document_sha256"]:
            raise ValueError("Mapper calibration document fingerprint differs")
        result.update(shingles(words(context)))
    return result


def paper_text(paper):
    sections = ["\n".join([section["section_name"] or "", *section["paragraphs"]])
                for section in paper["full_text"]]
    return "\n\n".join(part.strip() for part in [paper["title"], paper["abstract"], *sections]
                       if part and part.strip())


def evidenced_questions(paper, terms):
    """Keep distinct questions with at least one answer evidenced in the supplied text."""
    normalized = f" {' '.join(terms)} "
    seen = set()
    for item in paper["qas"]:
        question = item["question"].strip()
        key = tuple(words(question))
        if not key or key in seen:
            continue
        answers = []
        for annotation in item["answers"]:
            answer = annotation["answer"]
            if answer["unanswerable"]:
                continue
            has_evidence = False
            for evidence in answer["evidence"]:
                evidence_terms = words(evidence)
                if evidence_terms and f" {' '.join(evidence_terms)} " in normalized:
                    has_evidence = True
                    break
            if not has_evidence:
                continue
            yes_no = answer["yes_no"]
            text = ("yes" if yes_no else "no") if yes_no is not None else (
                " ".join(answer["extractive_spans"]) or answer["free_form_answer"])
            if text.strip():
                answers.append(text.strip())
        if answers:
            seen.add(key)
            yield item["question_id"], question, list(dict.fromkeys(answers))


def load_reuse_cohort(dataset, reference, profile, template, tokenizers, seed):
    if sha(dataset["path"]) != dataset["sha256"]:
        raise ValueError("QASPER test archive changed after configuration was read")
    check_tokenizers(template, tokenizers)
    excluded = calibration_shingles(reference)
    with tarfile.open(dataset["path"], "r:gz") as archive:
        stream = archive.extractfile("qasper-test-v0.3.json")
        if stream is None:
            raise ValueError("The QASPER archive is missing its test JSON")
        with stream:
            papers = json.load(stream)
    paper_ids = sorted(papers)
    Random(seed).shuffle(paper_ids)
    wanted = profile["documents"]
    questions_per_doc = profile["questions_per_document"]
    minimum, maximum = profile["prefix_tokens"]
    if wanted < 1 or questions_per_doc < 1 or not 0 < minimum <= maximum:
        raise ValueError("Invalid QASPER cohort size or prefix token range")

    cohort, prompts, seen_prefixes = [], [], set()
    for paper_id in paper_ids:
        paper = papers[paper_id]
        context = paper_text(paper)
        terms = words(context)
        if any(part in excluded for part in shingles(terms)):
            continue
        questions = list(evidenced_questions(paper, terms))[:questions_per_doc]
        if len(questions) < questions_per_doc:
            continue
        first_question = questions[0][1]
        first = render_qa_prompt(context, first_question, template, tokenizers)
        ids, boundary = first
        if not minimum <= boundary <= maximum:
            continue
        prefix = tuple(ids[:boundary])
        if prefix in seen_prefixes:
            continue
        selected = [first, *(render_qa_prompt(context, question, template, tokenizers)
                             for _, question, _ in questions[1:])]
        if any(tuple(ids[:offset]) != prefix for ids, offset in selected):
            continue
        if len({tuple(ids) for ids, _ in selected}) != questions_per_doc:
            continue

        seen_prefixes.add(prefix)
        digest = hashlib.sha256(context.encode()).hexdigest()
        for (question_id, _, answers), (ids, offset) in zip(questions, selected, strict=True):
            cohort.append({
                "dataset": "qasper_v03_test", "paper_id": paper_id, "question_id": question_id,
                "group": "qasper_v03_test", "document_sha256": digest, "token_sha256": token_hash(ids),
                "prompt_tokens": len(ids), "source_prefix_tokens": offset, "answers": answers,
            })
        prompts.extend(selected)
        if len(seen_prefixes) == wanted:
            return cohort, prompts
    raise ValueError(f"Found {len(seen_prefixes)} eligible QASPER papers; profile requires {wanted}")
