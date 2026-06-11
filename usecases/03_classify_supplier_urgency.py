"""
Use case: Tag inbound supplier emails by intent / urgency.
Travel-retail relevance: Procurement teams handle dispute, delivery, invoice and routine emails in one inbox.
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

USE_CASE_ID = "03_classify_supplier_urgency"
MODEL_ID = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
TASK_TYPE = "zero_shot_classification"

CANDIDATE_LABELS = ["urgent", "routine", "dispute", "delivery_update", "invoice_query"]

TEST_CASES = [
    {"input": "PO #12345 was overcharged by $500, please refund", "expected": "dispute"},
    {"input": "Please find attached the monthly catalogue update", "expected": "routine"},
    {"input": "Container delayed at customs, ETA pushed two weeks", "expected": "delivery_update"},
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
        description="Tag inbound supplier emails by intent / urgency.",
        domain_relevance="Procurement teams handle dispute, delivery, invoice and routine emails in one inbox.",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
