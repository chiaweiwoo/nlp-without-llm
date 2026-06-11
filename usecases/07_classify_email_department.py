"""
Formal problem: Multiclass Text Classification.
Luxury-retail application: Route operational incidents with keyword-based rules.
Model: curated_luxury_retail_rules_v1
Method: Keyword rules
"""

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import load_json_fixture

USE_CASE_ID = "07_classify_email_department"
MODEL_ID = "curated_luxury_retail_rules_v1"
TASK_TYPE = "multiclass_classification"
PROBLEM_NAME = "Multiclass Text Classification"
TECHNIQUE_NAME = "Keyword rules"
APPLICATION_NAME = "Route luxury-retail operational incidents into inventory, client service, or loss prevention."
COMPARISON_GROUP = "incident_routing"
RUNTIME_TIER = "fast"

TEST_CASES = load_json_fixture("multiclass_text_classification/examples.json")
CLASS_KEYWORDS = {
    "inventory_exception": {"stock", "inventory", "available", "shelf", "shipment", "storeroom"},
    "client_service_exception": {"waited", "assistance", "traveller", "customer", "collection", "counter"},
    "loss_prevention_exception": {"missing", "security", "bracelet", "count", "locked", "investigating"},
}


def classify_with_rules(text: str) -> tuple[str, dict]:
    lowered = text.lower()
    scores = {}
    for label, keywords in CLASS_KEYWORDS.items():
        scores[label] = sum(1 for keyword in keywords if keyword in lowered)
    actual = max(scores, key=scores.get)
    return actual, scores


def run() -> dict:
    t0 = time.perf_counter()
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        actual, scores = classify_with_rules(tc["input"])
        passed = actual == tc["expected"]
        score_notes = ", ".join(f"{label}:{value}" for label, value in scores.items())
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
        description="Route luxury-retail operational incidents with keyword-based rules.",
        domain_relevance="A deterministic rules baseline reveals where simple routing is already sufficient.",
        model=MODEL_ID,
        library="custom",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
        problem_name=PROBLEM_NAME,
        technique_name=TECHNIQUE_NAME,
        application_name=APPLICATION_NAME,
        comparison_group=COMPARISON_GROUP,
        runtime_tier=RUNTIME_TIER,
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
