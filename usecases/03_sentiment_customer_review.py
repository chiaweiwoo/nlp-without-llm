"""
Formal problem: Sentiment Analysis.
Luxury-retail application: Score boutique review sentiment with a compact lexicon baseline.
Model: curated_luxury_retail_lexicon_v1
Method: Lexicon scoring
"""

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import lexicon_sentiment, load_json_fixture

USE_CASE_ID = "03_sentiment_customer_review"
MODEL_ID = "curated_luxury_retail_lexicon_v1"
TASK_TYPE = "sentiment"
PROBLEM_NAME = "Sentiment Analysis"
TECHNIQUE_NAME = "Lexicon scoring"
APPLICATION_NAME = "Evaluate post-purchase luxury-retail reviews with a transparent lexicon baseline."
COMPARISON_GROUP = "sentiment_analysis"
RUNTIME_TIER = "fast"

TEST_CASES = load_json_fixture("sentiment_analysis/examples.json")
POSITIVE_TERMS = {
    "beautifully",
    "care",
    "helpful",
    "excellent",
    "polite",
    "smooth",
    "luxurious",
    "impressed",
}
NEGATIVE_TERMS = {
    "damaged",
    "refused",
    "late",
    "broken",
    "rude",
    "missing",
    "issue",
    "delay",
}


def run() -> dict:
    t0 = time.perf_counter()
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        actual, debug = lexicon_sentiment(tc["input"], POSITIVE_TERMS, NEGATIVE_TERMS)
        passed = actual == tc["expected"]
        results.append(
            {
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": (
                    f"positive_hits={debug['positive_hits']} "
                    f"negative_hits={debug['negative_hits']} "
                    f"token_count={debug['token_count']}"
                ),
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Score luxury-retail customer review sentiment with a transparent domain lexicon.",
        domain_relevance="A simple baseline helps measure whether a small local transformer is worth the extra cost.",
        model=MODEL_ID,
        library="custom",
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
