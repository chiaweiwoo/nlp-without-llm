"""
Formal problem: Multiclass Text Classification.
Luxury-retail application: Route operational incidents with few-shot prototype classification.
Model: sentence-transformers/all-mpnet-base-v2
Method: Few-shot embedding prototypes
"""

import json
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import load_json_fixture

USE_CASE_ID = "09_classify_feedback_topic"
MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
TASK_TYPE = "multiclass_classification"
PROBLEM_NAME = "Multiclass Text Classification"
TECHNIQUE_NAME = "Few-shot embedding prototypes"
APPLICATION_NAME = "Route luxury-retail operational incidents into inventory, client service, or loss prevention."
COMPARISON_GROUP = "incident_routing"
RUNTIME_TIER = "medium"

TEST_CASES = load_json_fixture("multiclass_text_classification/examples.json")
SUPPORT_EXAMPLES = load_json_fixture("multiclass_text_classification/support_examples.json")


def build_class_prototypes(model: SentenceTransformer) -> dict[str, torch.Tensor]:
    prototypes = {}
    for label, examples in SUPPORT_EXAMPLES.items():
        embeddings = model.encode(examples, convert_to_tensor=True)
        prototypes[label] = embeddings.mean(dim=0)
    return prototypes


def run() -> dict:
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_ID, device="cpu")
    prototypes = build_class_prototypes(model)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        query_embedding = model.encode(tc["input"], convert_to_tensor=True)
        scores = {
            label: round(util.cos_sim(query_embedding, prototype).item(), 4)
            for label, prototype in prototypes.items()
        }
        actual = max(scores, key=scores.get)
        passed = actual == tc["expected"]
        score_notes = ", ".join(
            f"{label}:{score}" for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )
        results.append(
            {
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"prototype_cosine=[{score_notes}]",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Route luxury-retail operational incidents with few-shot prototype classification over embeddings.",
        domain_relevance="This compares a lightweight labelled-support approach against both rules and zero-shot NLI.",
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
