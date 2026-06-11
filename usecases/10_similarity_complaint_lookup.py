"""
Formal problem: Information Retrieval.
Luxury-retail application: Retrieve SOP passages with BM25 lexical search.
Model: curated_bm25_retriever_v1
Method: BM25 lexical retrieval
"""

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import bm25_scores, build_bm25_index, load_json_fixture

USE_CASE_ID = "10_similarity_complaint_lookup"
MODEL_ID = "curated_bm25_retriever_v1"
TASK_TYPE = "information_retrieval"
PROBLEM_NAME = "Information Retrieval"
TECHNIQUE_NAME = "BM25 lexical retrieval"
APPLICATION_NAME = "Retrieve luxury-retail SOP passages for staff questions using lexical matching."
COMPARISON_GROUP = "sop_retrieval"
RUNTIME_TIER = "fast"

CORPUS = load_json_fixture("information_retrieval/corpus.json")
TEST_CASES = load_json_fixture("information_retrieval/examples.json")


def run() -> dict:
    t0 = time.perf_counter()
    index = build_bm25_index([item["text"] for item in CORPUS])
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        scores = bm25_scores(tc["input"], index)
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        actual = CORPUS[best_idx]["id"]
        passed = actual == tc["expected"]
        score_notes = ", ".join(f"{item['id']}:{scores[idx]}" for idx, item in enumerate(CORPUS))
        results.append(
            {
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"bm25_scores=[{score_notes}] matched_text='{CORPUS[best_idx]['text']}'",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Retrieve luxury-retail SOP passages with BM25 lexical search.",
        domain_relevance="This baseline shows how far classical lexical retrieval goes before embeddings or reranking are needed.",
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
