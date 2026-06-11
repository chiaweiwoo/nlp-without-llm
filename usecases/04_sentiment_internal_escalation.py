"""
Formal problem: Sentiment Analysis.
Luxury-retail application: Score boutique review sentiment with a transformer classifier.
Model: cardiffnlp/twitter-roberta-base-sentiment-latest
Method: Transformer sequence classification
"""

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import pipeline

from lab.contract import build_result
from lab.study_utils import load_json_fixture, normalize_sentiment_label

USE_CASE_ID = "04_sentiment_internal_escalation"
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
TASK_TYPE = "sentiment"
PROBLEM_NAME = "Sentiment Analysis"
TECHNIQUE_NAME = "Transformer sequence classification"
APPLICATION_NAME = "Evaluate post-purchase luxury-retail reviews with a pretrained transformer classifier."
COMPARISON_GROUP = "sentiment_analysis"
RUNTIME_TIER = "medium"

TEST_CASES = load_json_fixture("sentiment_analysis/examples.json")


def run() -> dict:
    t0 = time.perf_counter()
    classifier = pipeline("sentiment-analysis", model=MODEL_ID)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        out = classifier(tc["input"])[0]
        actual = normalize_sentiment_label(out["label"])
        passed = actual == tc["expected"]
        results.append(
            {
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"raw_label={out['label']} score={round(out['score'], 4)}",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Score luxury-retail customer review sentiment with a pretrained transformer classifier.",
        domain_relevance="This provides a higher-capability local baseline to compare against the lexicon method.",
        model=MODEL_ID,
        library="transformers",
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
