"""
Use case: Detect which language a customer message is in.
Travel-retail relevance: Airport customer base is multilingual; route to right-language support.
Model: papluca/xlm-roberta-base-language-detection
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

USE_CASE_ID = "10_language_detection"
MODEL_ID = "papluca/xlm-roberta-base-language-detection"
TASK_TYPE = "language_detection"

TEST_CASES = [
    {"input": "我在新加坡买了一瓶酒", "expected": "zh"},
    {"input": "ありがとうございました", "expected": "ja"},
    {"input": "I want to claim my tax refund", "expected": "en"},
]

def run() -> dict:
    t0 = time.perf_counter()
    classifier = pipeline("text-classification", model=MODEL_ID)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        out = classifier(tc["input"])[0]
        actual = out["label"].lower()
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
        description="Detect which language a customer message is in.",
        domain_relevance="Airport customer base is multilingual; route to right-language support.",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    sys.stdout.buffer.write(json.dumps(run(), indent=2, ensure_ascii=False).encode('utf-8'))
    print()
