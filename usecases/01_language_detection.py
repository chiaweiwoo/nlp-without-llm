"""
Formal problem: Language Identification.
Luxury-retail application: Identify the language of traveller enquiries before support triage.
Model: papluca/xlm-roberta-base-language-detection
Method: Transformer classification
"""

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import pipeline

from lab.contract import build_result
from lab.study_utils import load_json_fixture

USE_CASE_ID = "01_language_detection"
MODEL_ID = "papluca/xlm-roberta-base-language-detection"
TASK_TYPE = "language_detection"
PROBLEM_NAME = "Language Identification"
TECHNIQUE_NAME = "Transformer classification"
APPLICATION_NAME = "Identify the language of traveller enquiries for luxury-retail support triage."
COMPARISON_GROUP = "language_identification"
RUNTIME_TIER = "fast"

TEST_CASES = load_json_fixture("language_identification/examples.json")


def run() -> dict:
    t0 = time.perf_counter()
    classifier = pipeline("text-classification", model=MODEL_ID)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        out = classifier(tc["input"])[0]
        actual = out["label"].lower()
        passed = actual == tc["expected"]
        results.append(
            {
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"raw_label={out['label']} score={round(out['score'], 4)}",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Identify the language of traveller enquiries before routing them to support.",
        domain_relevance="Luxury-retail support teams receive enquiries from international travellers before departure.",
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
