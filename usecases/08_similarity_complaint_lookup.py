"""
Use case: Given a new complaint, retrieve the most similar past complaint.
Travel-retail relevance: Customer service can reuse previous resolutions.
Model: sentence-transformers/all-MiniLM-L6-v2
Library: sentence-transformers
"""

import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer, util
from lab.contract import build_result

USE_CASE_ID = "08_similarity_complaint_lookup"
MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
TASK_TYPE = "semantic_similarity"

CORPUS = [
    {"id": "c1", "text": "Got a damaged cosmetic item"},
    {"id": "c2", "text": "Cashier was rude at the HKG store"},
    {"id": "c3", "text": "Double-charged on my credit card"},
    {"id": "c4", "text": "Forgot to claim tax refund at the airport"},
    {"id": "c5", "text": "Item different from website description"},
    {"id": "c6", "text": "Long queue at checkout during peak hours"},
]

TEST_CASES = [
    {"input": "My perfume bottle arrived broken", "expected": "c1"},
    {"input": "Staff was impolite at HKG store", "expected": "c2"},
    {"input": "Charged twice for the same purchase", "expected": "c3"},
]

def run() -> dict:
    t0 = time.perf_counter()
    # Force CPU to comply with constraints
    model = SentenceTransformer(MODEL_ID, device="cpu")
    
    # Pre-encode corpus
    corpus_texts = [item["text"] for item in CORPUS]
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        query_embedding = model.encode(tc["input"], convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        
        # Get index of top-1 match
        best_idx = cos_scores.argmax().item()
        actual = CORPUS[best_idx]["id"]
        passed = (actual == tc["expected"])
        
        # Build score notes
        scores_summary = ", ".join(f"{item['id']}:{round(cos_scores[i].item(), 3)}" for i, item in enumerate(CORPUS))
        
        results.append({
            "input": tc["input"],
            "expected": tc["expected"],
            "actual": actual,
            "passed": passed,
            "inference_time_s": round(time.perf_counter() - t1, 4),
            "notes": f"cosine_scores=[{scores_summary}] matched_text='{CORPUS[best_idx]['text']}'",
        })

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Given a new complaint, retrieve the most similar past complaint.",
        domain_relevance="Customer service can reuse previous resolutions.",
        model=MODEL_ID,
        library="sentence-transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
