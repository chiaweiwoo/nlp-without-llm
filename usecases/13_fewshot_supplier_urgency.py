"""
Formal problem: Extractive Question Answering.
Luxury-retail application: Extract exact policy answers from staff reference text.
Model: deepset/tinyroberta-squad2
Method: Transformer span extraction
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import load_json_fixture, normalize_text

USE_CASE_ID = "13_fewshot_supplier_urgency"
MODEL_ID = "deepset/tinyroberta-squad2"
TASK_TYPE = "extractive_qa"
PROBLEM_NAME = "Extractive Question Answering"
TECHNIQUE_NAME = "Transformer span extraction"
APPLICATION_NAME = "Extract exact policy answers from luxury-retail SOP text."
COMPARISON_GROUP = "extractive_question_answering"
RUNTIME_TIER = "medium"

TEST_CASES = load_json_fixture("extractive_question_answering/examples.json")


def run() -> dict:
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        inputs = tokenizer(
            tc["question"],
            tc["context"],
            return_tensors="pt",
            truncation=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        start_idx = int(torch.argmax(outputs.start_logits))
        end_idx = int(torch.argmax(outputs.end_logits))
        if end_idx < start_idx:
            end_idx = start_idx
        answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
        actual = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        passed = normalize_text(actual) == normalize_text(tc["expected"])
        results.append(
            {
                "input": tc["question"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": (
                    f"start_idx={start_idx} end_idx={end_idx} "
                    f"context_length={len(tc['context'])}"
                ),
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Extract exact answers from luxury-retail policy passages.",
        domain_relevance="Staff often need the exact rule from a policy, not just a similar document match.",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
        problem_name=PROBLEM_NAME,
        technique_name=TECHNIQUE_NAME,
        application_name=APPLICATION_NAME,
        comparison_group=COMPARISON_GROUP,
        runtime_tier=RUNTIME_TIER,
        primary_metric="exact_match",
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
