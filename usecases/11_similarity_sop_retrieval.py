"""
Formal problem: Information Retrieval.
Luxury-retail application: Retrieve SOP passages with dense bi-encoder embeddings.
Model: sentence-transformers/all-mpnet-base-v2
Method: Dense bi-encoder retrieval
"""

import json
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import load_json_fixture

USE_CASE_ID = "11_similarity_sop_retrieval"
MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
TASK_TYPE = "information_retrieval"
PROBLEM_NAME = "Information Retrieval"
TECHNIQUE_NAME = "Dense bi-encoder retrieval"
APPLICATION_NAME = "Retrieve luxury-retail SOP passages for staff questions using semantic embeddings."
COMPARISON_GROUP = "sop_retrieval"
RUNTIME_TIER = "medium"

CORPUS = load_json_fixture("information_retrieval/corpus.json")
TEST_CASES = load_json_fixture("information_retrieval/examples.json")


def run() -> dict:
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_ID, device="cpu")
    corpus_embeddings = model.encode([item["text"] for item in CORPUS], convert_to_tensor=True)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        query_embedding = model.encode(tc["input"], convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        best_idx = cos_scores.argmax().item()
        actual = CORPUS[best_idx]["id"]
        passed = actual == tc["expected"]
        score_notes = ", ".join(
            f"{item['id']}:{round(cos_scores[idx].item(), 3)}"
            for idx, item in enumerate(CORPUS)
        )
        results.append(
            {
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"cosine_scores=[{score_notes}] matched_text='{CORPUS[best_idx]['text']}'",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Retrieve luxury-retail SOP passages with dense bi-encoder embeddings.",
        domain_relevance="This measures the semantic retrieval gain from local embeddings over purely lexical search.",
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
