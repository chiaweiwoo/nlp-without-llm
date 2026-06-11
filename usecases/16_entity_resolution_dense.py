"""
Formal problem: Entity Resolution.
Luxury-retail application: Resolve informal product names with dense catalogue matching.
Model: sentence-transformers/all-mpnet-base-v2
Method: Dense catalogue matching
"""

import json
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import load_json_fixture

USE_CASE_ID = "16_entity_resolution_dense"
MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
TASK_TYPE = "entity_resolution"
PROBLEM_NAME = "Entity Resolution"
TECHNIQUE_NAME = "Dense catalogue matching"
APPLICATION_NAME = "Resolve informal luxury-retail product names to canonical catalogue SKUs with embeddings."
COMPARISON_GROUP = "entity_resolution"
RUNTIME_TIER = "medium"

CATALOG = load_json_fixture("entity_resolution/catalog.json")
TEST_CASES = load_json_fixture("entity_resolution/examples.json")


def run() -> dict:
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_ID, device="cpu")
    catalog_embeddings = model.encode([item["name"] for item in CATALOG], convert_to_tensor=True)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        query_embedding = model.encode(tc["input"], convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, catalog_embeddings)[0]
        best_idx = cos_scores.argmax().item()
        actual = CATALOG[best_idx]["sku"]
        passed = actual == tc["expected"]
        notes = ", ".join(
            f"{item['sku']}:{round(cos_scores[idx].item(), 3)}"
            for idx, item in enumerate(CATALOG[:3])
        )
        results.append(
            {
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"sample_scores=[{notes}]",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Resolve informal luxury-retail product names with dense catalogue matching.",
        domain_relevance="This measures whether embeddings improve robustness for aliases, abbreviations, and misspellings.",
        model=MODEL_ID,
        library="sentence-transformers",
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
