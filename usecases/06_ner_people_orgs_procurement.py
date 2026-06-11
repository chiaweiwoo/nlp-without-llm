"""
Use case: Pull people and organisation names from procurement emails.
Travel-retail relevance: Auto-populate CRM and supplier contact records.
Model: dslim/bert-base-NER
Library: transformers
"""

import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import pipeline
from lab.contract import build_result

USE_CASE_ID = "06_ner_people_orgs_procurement"
MODEL_ID = "dslim/bert-base-NER"
TASK_TYPE = "ner"

TEST_CASES = [
    {
        "input": "John from Acme Logistics confirmed delivery",
        "expected": [["John", "PER"], ["Acme Logistics", "ORG"]],
    },
    {
        "input": "Please contact Sarah Tan at Globex regarding the new collection",
        "expected": [["Sarah Tan", "PER"], ["Globex", "ORG"]],
    },
    {
        "input": "Diageo representative will visit our Singapore office Monday",
        "expected": [["Diageo", "ORG"], ["Singapore", "LOC"]],
    },
]

def calculate_f1(expected_set: set, actual_set: set) -> tuple[float, float, float]:
    if not expected_set and not actual_set:
        return 1.0, 1.0, 1.0
    if not expected_set or not actual_set:
        return 0.0, 0.0, 0.0
        
    intersection = expected_set.intersection(actual_set)
    precision = len(intersection) / len(actual_set)
    recall = len(intersection) / len(expected_set)
    
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
        
    f1 = 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)

def run() -> dict:
    t0 = time.perf_counter()
    ner = pipeline("ner", model=MODEL_ID, aggregation_strategy="simple")
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        out = ner(tc["input"])
        
        # Extract actual entities as a set of (word, entity_group)
        actual_entities = {(item["word"].strip(), item["entity_group"]) for item in out}
        expected_entities = {(item[0], item[1]) for item in tc["expected"]}
        
        precision, recall, f1 = calculate_f1(expected_entities, actual_entities)
        passed = (f1 >= 0.66)
        
        actual_list = sorted(list(actual_entities))
        expected_list = sorted(list(expected_entities))
        
        results.append({
            "input": tc["input"],
            "expected": expected_list,
            "actual": actual_list,
            "passed": passed,
            "inference_time_s": round(time.perf_counter() - t1, 4),
            "notes": f"precision={precision} recall={recall} f1={f1} raw_ner={str([(item['word'], item['entity_group']) for item in out])}",
        })

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Pull people and organisation names from procurement emails.",
        domain_relevance="Auto-populate CRM and supplier contact records.",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
