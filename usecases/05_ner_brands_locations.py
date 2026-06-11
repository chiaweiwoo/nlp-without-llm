"""
Formal problem: Named Entity Recognition.
Luxury-retail application: Extract brands, people, organisations, and locations with a gazetteer baseline.
Model: curated_luxury_retail_gazetteer_v1
Method: Gazetteer and phrase matching
"""

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import calculate_entity_f1, load_json_fixture

USE_CASE_ID = "05_ner_brands_locations"
MODEL_ID = "curated_luxury_retail_gazetteer_v1"
TASK_TYPE = "ner"
PROBLEM_NAME = "Named Entity Recognition"
TECHNIQUE_NAME = "Gazetteer and phrase matching"
APPLICATION_NAME = "Extract brands, people, organisations, and locations from luxury-retail service notes."
COMPARISON_GROUP = "named_entity_recognition"
RUNTIME_TIER = "fast"

TEST_CASES = load_json_fixture("named_entity_recognition/examples.json")
ENTITY_PHRASES = {
    ("Dior", "ORG"),
    ("Cartier", "ORG"),
    ("Macallan", "ORG"),
    ("Sophie Tan", "PER"),
    ("James Wu", "PER"),
    ("Elaine Lim", "PER"),
    ("Changi Terminal 1", "LOC"),
    ("Heathrow Boutique", "ORG"),
    ("Hong Kong Airport", "LOC"),
}


def extract_entities(text: str) -> set[tuple[str, str]]:
    lowered = text.lower()
    found = set()
    for phrase, label in ENTITY_PHRASES:
        if phrase.lower() in lowered:
            found.add((phrase, label))
    return found


def run() -> dict:
    t0 = time.perf_counter()
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        actual_entities = extract_entities(tc["input"])
        expected_entities = {(entity[0], entity[1]) for entity in tc["expected"]}
        precision, recall, f1 = calculate_entity_f1(expected_entities, actual_entities)
        results.append(
            {
                "input": tc["input"],
                "expected": sorted(list(expected_entities)),
                "actual": sorted(list(actual_entities)),
                "passed": f1 >= 0.66,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"precision={precision} recall={recall} f1={f1}",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Extract brands, people, organisations, and locations with a luxury-retail gazetteer baseline.",
        domain_relevance="A deterministic baseline shows how far curated phrase matching can go before a transformer is needed.",
        model=MODEL_ID,
        library="custom",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
        problem_name=PROBLEM_NAME,
        technique_name=TECHNIQUE_NAME,
        application_name=APPLICATION_NAME,
        comparison_group=COMPARISON_GROUP,
        runtime_tier=RUNTIME_TIER,
        primary_metric="entity_f1_threshold",
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
