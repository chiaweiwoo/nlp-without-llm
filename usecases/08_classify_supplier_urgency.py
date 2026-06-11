"""
Formal problem: Multiclass Text Classification.
Luxury-retail application: Route operational incidents with zero-shot NLI.
Model: MoritzLaurer/deberta-v3-base-zeroshot-v2.0
Method: Zero-shot NLI classification
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import load_json_fixture

USE_CASE_ID = "08_classify_supplier_urgency"
MODEL_ID = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
TASK_TYPE = "multiclass_classification"
PROBLEM_NAME = "Multiclass Text Classification"
TECHNIQUE_NAME = "Zero-shot NLI classification"
APPLICATION_NAME = "Route luxury-retail operational incidents into inventory, client service, or loss prevention."
COMPARISON_GROUP = "incident_routing"
RUNTIME_TIER = "medium"

TEST_CASES = load_json_fixture("multiclass_text_classification/examples.json")
HYPOTHESES = {
    "inventory_exception": "This incident is about missing stock, unavailable inventory, or a failed collection because the item is not physically available.",
    "client_service_exception": "This incident is about delayed assistance, poor customer handling, or a service failure during luxury retail collection.",
    "loss_prevention_exception": "This incident is about shrinkage, a missing high-value item, or a security investigation after stock count.",
}


def run() -> dict:
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        scores = {}
        for label, hypothesis in HYPOTHESES.items():
            inputs = tokenizer(tc["input"], hypothesis, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            scores[label] = torch.softmax(outputs.logits, dim=-1)[0][0].item()

        actual = max(scores, key=scores.get)
        passed = actual == tc["expected"]
        score_notes = ", ".join(
            f"{label}:{round(score, 3)}"
            for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )
        results.append(
            {
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"scores=[{score_notes}]",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Route luxury-retail operational incidents with zero-shot NLI classification.",
        domain_relevance="This shows how a local NLI model handles incident routing without labelled task-specific training.",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
        problem_name=PROBLEM_NAME,
        technique_name=TECHNIQUE_NAME,
        application_name=APPLICATION_NAME,
        comparison_group=COMPARISON_GROUP,
        runtime_tier=RUNTIME_TIER,
    )


if __name__ == "__main__":
    sys.stdout.buffer.write(json.dumps(run(), indent=2, ensure_ascii=False).encode("utf-8"))
    print()
