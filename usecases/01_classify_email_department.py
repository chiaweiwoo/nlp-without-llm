"""
Use case: Route internal emails to the correct department.
Travel-retail relevance: Multi-department retail operations get hundreds of emails; auto-routing cuts triage time.
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

USE_CASE_ID = "01_classify_email_department"
MODEL_ID = "facebook/bart-large-mnli"
TASK_TYPE = "zero_shot_classification"

CANDIDATE_LABELS = ["IT", "Planner", "Pricing", "Warehouse", "Finance", "HR"]

TEST_CASES = [
    {"input": "Server is down, customers can't checkout", "expected": "IT"},
    {"input": "Need to reorder fragrance stock for the airport store, running low", "expected": "Planner"},
    {"input": "Q3 expense report due Friday", "expected": "Finance"},
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
        description="Route incoming internal emails to the correct department.",
        domain_relevance="Reduces manual triage in a multi-department retail org.",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
