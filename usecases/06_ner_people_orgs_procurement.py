"""
Formal problem: Named Entity Recognition.
Luxury-retail application: Extract brands, people, organisations, and locations with a transformer tagger.
Model: dslim/bert-base-NER
Method: Transformer token classification
"""

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import pipeline

from lab.contract import build_result
from lab.study_utils import calculate_entity_f1, load_json_fixture

USE_CASE_ID = "06_ner_people_orgs_procurement"
MODEL_ID = "dslim/bert-base-NER"
TASK_TYPE = "ner"
PROBLEM_NAME = "Named Entity Recognition"
TECHNIQUE_NAME = "Transformer token classification"
APPLICATION_NAME = "Extract brands, people, organisations, and locations from luxury-retail service notes."
COMPARISON_GROUP = "named_entity_recognition"
RUNTIME_TIER = "medium"

TEST_CASES = load_json_fixture("named_entity_recognition/examples.json")


def run() -> dict:
    t0 = time.perf_counter()
    ner = pipeline("ner", model=MODEL_ID, aggregation_strategy="simple")
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        out = ner(tc["input"])
        actual_entities = {(item["word"].strip(), item["entity_group"]) for item in out}
        expected_entities = {(entity[0], entity[1]) for entity in tc["expected"]}
        precision, recall, f1 = calculate_entity_f1(expected_entities, actual_entities)
        results.append(
            {
                "input": tc["input"],
                "expected": sorted(list(expected_entities)),
                "actual": sorted(list(actual_entities)),
                "passed": f1 >= 0.66,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": (
                    f"precision={precision} recall={recall} f1={f1} "
                    f"raw_ner={str([(item['word'], item['entity_group']) for item in out])}"
                ),
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Extract brands, people, organisations, and locations with a pretrained transformer tagger.",
        domain_relevance="This measures the accuracy gain from a general transformer NER model on luxury-retail notes.",
        model=MODEL_ID,
        library="transformers",
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
