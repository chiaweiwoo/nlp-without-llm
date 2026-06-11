"""
Formal problem: Template-Based Information Extraction.
Luxury-retail application: Parse receipt and stock-transfer text with deterministic rules.
Model: curated_regex_parser_v1
Method: Regular expressions and deterministic parsing
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import load_json_fixture

USE_CASE_ID = "14_fewshot_feedback_topic"
MODEL_ID = "curated_regex_parser_v1"
TASK_TYPE = "information_extraction"
PROBLEM_NAME = "Template-Based Information Extraction"
TECHNIQUE_NAME = "Regular expressions and deterministic parsing"
APPLICATION_NAME = "Parse luxury-retail receipt and stock-transfer text into structured fields."
COMPARISON_GROUP = "template_information_extraction"
RUNTIME_TIER = "fast"

TEST_CASES = load_json_fixture("template_based_information_extraction/examples.json")


def extract_fields(text: str) -> dict:
    result: dict = {}
    if text.startswith("Transfer"):
        match = re.search(
            r"Transfer (\d+) x SKU ([A-Z0-9-]+) at ([A-Z]{3}) (\d+) from (.+) to (.+)\.",
            text,
        )
        if match:
            result = {
                "quantity": int(match.group(1)),
                "sku": match.group(2),
                "currency": match.group(3),
                "price": int(match.group(4)),
                "from_location": match.group(5),
                "to_location": match.group(6),
            }
    elif text.startswith("Receipt"):
        match = re.search(
            r"Receipt ([A-Z0-9-]+): (\d+) (.+), total ([A-Z]{3}) (\d+)\.",
            text,
        )
        if match:
            result = {
                "receipt_id": match.group(1),
                "quantity": int(match.group(2)),
                "product": match.group(3),
                "currency": match.group(4),
                "total": int(match.group(5)),
            }
    elif text.startswith("Received"):
        match = re.search(
            r"Received (\d+) of (\d+) (.+) at (.+)\.",
            text,
        )
        if match:
            result = {
                "received_quantity": int(match.group(1)),
                "expected_quantity": int(match.group(2)),
                "product": match.group(3),
                "location": match.group(4),
            }
    return result


def run() -> dict:
    t0 = time.perf_counter()
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        actual = extract_fields(tc["input"])
        passed = actual == tc["expected"]
        results.append(
            {
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"field_count={len(actual)}",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Parse luxury-retail receipt and stock-transfer text with deterministic rules.",
        domain_relevance="Many operational text formats are regular enough that local parsing beats heavier models on speed and reliability.",
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
