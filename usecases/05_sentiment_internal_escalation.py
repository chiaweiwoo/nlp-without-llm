"""
Use case: Flag internal emails with negative tone as escalation candidates.
Travel-retail relevance: Catch operational frustration before it becomes a manager escalation.
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

USE_CASE_ID = "05_sentiment_internal_escalation"
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
TASK_TYPE = "sentiment"

TEST_CASES = [
    {"input": "Quick question on SKU mapping when you have time", "expected": "not_escalated"},
    {"input": "Third time I'm raising this, no one is responding", "expected": "escalated"},
    {"input": "Update: warehouse received the shipment", "expected": "not_escalated"},
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
    t0 = time.perf_counter()
    classifier = pipeline("sentiment-analysis", model=MODEL_ID)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        out = classifier(tc["input"])[0]
        norm = normalise_label(out["label"])
        
        # Mapping: negative -> escalated; positive/neutral -> not_escalated
        actual = "escalated" if norm == "negative" else "not_escalated"
        passed = (actual == tc["expected"])
        
        results.append({
            "input": tc["input"],
            "expected": tc["expected"],
            "actual": actual,
            "passed": passed,
            "inference_time_s": round(time.perf_counter() - t1, 4),
            "notes": f"raw_label={out['label']} normalised_sentiment={norm} score={round(out['score'], 4)}",
        })

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Flag internal emails with negative tone as escalation candidates.",
        domain_relevance="Catch operational frustration before it becomes a manager escalation.",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
