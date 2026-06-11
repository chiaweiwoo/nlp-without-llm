"""
Formal problem: Extractive Summarization.
Luxury-retail application: Select key sentences from shift-handover notes.
Model: curated_sentence_ranker_v1
Method: Sentence ranking
"""

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import load_json_fixture, rank_sentences_by_keywords

USE_CASE_ID = "18_extractive_summarization"
MODEL_ID = "curated_sentence_ranker_v1"
TASK_TYPE = "extractive_summarization"
PROBLEM_NAME = "Extractive Summarization"
TECHNIQUE_NAME = "Sentence ranking"
APPLICATION_NAME = "Select the most important sentences from luxury-retail handover notes."
COMPARISON_GROUP = "extractive_summarization"
RUNTIME_TIER = "fast"

TEST_CASES = load_json_fixture("extractive_summarization/examples.json")
PRIORITY_TERMS = {
    "vip",
    "collect",
    "customer",
    "preorder",
    "offline",
    "engineering",
    "security",
    "missing",
    "cctv",
    "locked",
    "sales",
}


def run() -> dict:
    t0 = time.perf_counter()
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        actual = rank_sentences_by_keywords(tc["input"], PRIORITY_TERMS, top_k=2)
        passed = actual == tc["expected"]
        results.append(
            {
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"selected_sentence_count={len(actual)}",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Select the key sentences from luxury-retail handover notes.",
        domain_relevance="Operations teams often need a concise handover without generating new text.",
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
