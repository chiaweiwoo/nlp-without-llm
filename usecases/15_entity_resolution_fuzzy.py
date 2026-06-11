"""
Formal problem: Entity Resolution.
Luxury-retail application: Resolve informal product names with fuzzy string matching.
Model: curated_fuzzy_matcher_v1
Method: String normalization and fuzzy matching
"""

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import fuzzy_ratio, load_json_fixture

USE_CASE_ID = "15_entity_resolution_fuzzy"
MODEL_ID = "curated_fuzzy_matcher_v1"
TASK_TYPE = "entity_resolution"
PROBLEM_NAME = "Entity Resolution"
TECHNIQUE_NAME = "String normalization and fuzzy matching"
APPLICATION_NAME = "Resolve informal luxury-retail product names to canonical catalogue SKUs."
COMPARISON_GROUP = "entity_resolution"
RUNTIME_TIER = "fast"

CATALOG = load_json_fixture("entity_resolution/catalog.json")
TEST_CASES = load_json_fixture("entity_resolution/examples.json")


def resolve_product(query: str) -> tuple[str, list[tuple[str, float]]]:
    scored = []
    for item in CATALOG:
        score = fuzzy_ratio(query, item["name"])
        scored.append((item["sku"], score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[0][0], scored


def run() -> dict:
    t0 = time.perf_counter()
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        actual, scored = resolve_product(tc["input"])
        passed = actual == tc["expected"]
        notes = ", ".join(f"{sku}:{score}" for sku, score in scored[:3])
        results.append(
            {
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"top_matches=[{notes}]",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Resolve informal luxury-retail product names with fuzzy string matching.",
        domain_relevance="This provides a lightweight baseline for catalogue normalization without embeddings.",
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
