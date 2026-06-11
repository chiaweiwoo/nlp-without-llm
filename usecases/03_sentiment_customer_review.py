"""
Use case: Classify customer reviews as positive, neutral, or negative.
Travel-retail relevance: Volume of post-purchase reviews makes manual sentiment scoring impossible.
Model: cardiffnlp/twitter-roberta-base-sentiment-latest
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

USE_CASE_ID = "03_sentiment_customer_review"
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
TASK_TYPE = "sentiment"

TEST_CASES = [
    {"input": "Amazing perfume selection, staff was so helpful!", "expected": "positive"},
    {"input": "It was okay, nothing special", "expected": "neutral"},
    {"input": "Waited 30 mins at checkout, never coming back", "expected": "negative"},
]

def normalise_label(label: str) -> str:
    lbl = label.lower()
    if "positive" in lbl or lbl == "label_2" or lbl == "pos":
        return "positive"
    if "negative" in lbl or lbl == "label_0" or lbl == "neg":
        return "negative"
    if "neutral" in lbl or lbl == "label_1" or lbl == "neu":
        return "neutral"
    return lbl

def run() -> dict:
    # 1. Time the model load
    t0 = time.perf_counter()
    classifier = pipeline("sentiment-analysis", model=MODEL_ID)
    load_time = time.perf_counter() - t0

    # 2. Run each test case, time the inference
    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        out = classifier(tc["input"])[0]
        actual = normalise_label(out["label"])
        passed = (actual == tc["expected"])
        results.append({
            "input": tc["input"],
            "expected": tc["expected"],
            "actual": actual,
            "passed": passed,
            "inference_time_s": round(time.perf_counter() - t1, 4),
            "notes": f"raw_label={out['label']} score={round(out['score'], 4)}",
        })

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Classify a customer review as positive / neutral / negative.",
        domain_relevance="Volume of post-purchase reviews makes manual sentiment scoring impossible.",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
