"""
Formal problem: Information Retrieval.
Luxury-retail application: Retrieve SOP passages with cross-encoder reranking.
Model: cross-encoder/ms-marco-MiniLM-L6-v2
Method: Cross-encoder reranking
"""

import json
import sys
import time
from pathlib import Path

from sentence_transformers import CrossEncoder

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import bm25_scores, build_bm25_index, load_json_fixture

USE_CASE_ID = "12_fewshot_email_department"
MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
TASK_TYPE = "information_retrieval"
PROBLEM_NAME = "Information Retrieval"
TECHNIQUE_NAME = "Cross-encoder reranking"
APPLICATION_NAME = "Retrieve luxury-retail SOP passages by reranking lexical candidates with a local cross-encoder."
COMPARISON_GROUP = "sop_retrieval"
RUNTIME_TIER = "medium"

CORPUS = load_json_fixture("information_retrieval/corpus.json")
TEST_CASES = load_json_fixture("information_retrieval/examples.json")
TOP_K = 3


def run() -> dict:
    t0 = time.perf_counter()
    lexical_index = build_bm25_index([item["text"] for item in CORPUS])
    reranker = CrossEncoder(MODEL_ID)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        lexical_scores = bm25_scores(tc["input"], lexical_index)
        candidate_indexes = sorted(
            range(len(lexical_scores)),
            key=lambda idx: lexical_scores[idx],
            reverse=True,
        )[:TOP_K]
        candidate_pairs = [(tc["input"], CORPUS[idx]["text"]) for idx in candidate_indexes]
        rerank_scores = reranker.predict(candidate_pairs)
        best_local_idx = max(range(len(candidate_indexes)), key=lambda idx: rerank_scores[idx])
        best_idx = candidate_indexes[best_local_idx]
        actual = CORPUS[best_idx]["id"]
        passed = actual == tc["expected"]

        score_notes = ", ".join(
            f"{CORPUS[idx]['id']}:{round(float(rerank_scores[pos]), 3)}"
            for pos, idx in enumerate(candidate_indexes)
        )
        results.append(
            {
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"rerank_scores=[{score_notes}] top_k={TOP_K}",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Retrieve luxury-retail SOP passages by reranking lexical candidates with a local cross-encoder.",
        domain_relevance="This shows the accuracy tradeoff when a slower but stronger local reranker is added after lexical retrieval.",
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
