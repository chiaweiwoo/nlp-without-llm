"""
Use case: Categorise customer feedback by topic.
Travel-retail relevance: Travel-retail customer experience teams triage thousands of post-purchase comments.
Model: facebook/bart-large-mnli
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

USE_CASE_ID = "02_classify_feedback_topic"
MODEL_ID = "facebook/bart-large-mnli"
TASK_TYPE = "zero_shot_classification"

CANDIDATE_LABELS = ["product_quality", "staff_service", "pricing", "store_experience", "checkout_process"]

TEST_CASES = [
    {"input": "The bottle of whisky I bought had a damaged seal", "expected": "product_quality"},
    {"input": "Cashier was rude when I asked for the tax-free form", "expected": "staff_service"},
    {"input": "Too expensive compared to my local market", "expected": "pricing"},
]

def run() -> dict:
    t0 = time.perf_counter()
    classifier = pipeline("zero-shot-classification", model=MODEL_ID)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        out = classifier(tc["input"], candidate_labels=CANDIDATE_LABELS)
        actual = out["labels"][0]
        passed = (actual == tc["expected"])
        
        # Build notes with full score distribution
        scores_summary = ", ".join(f"{lbl}:{round(score, 3)}" for lbl, score in zip(out["labels"], out["scores"]))
        
        results.append({
            "input": tc["input"],
            "expected": tc["expected"],
            "actual": actual,
            "passed": passed,
            "inference_time_s": round(time.perf_counter() - t1, 4),
            "notes": f"scores=[{scores_summary}]",
        })

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Categorise customer feedback by topic.",
        domain_relevance="Travel-retail customer experience teams triage thousands of post-purchase comments.",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
