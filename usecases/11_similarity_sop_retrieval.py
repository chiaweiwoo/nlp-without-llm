"""
Use case: Match a customer question to the most relevant internal SOP.
Travel-retail relevance: Front-line staff find the right policy quickly.
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

USE_CASE_ID = "11_similarity_sop_retrieval"
MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
TASK_TYPE = "semantic_similarity"

CORPUS = [
    {"id": "sop_returns", "text": "Return policy: items can be returned within 30 days with receipt"},
    {"id": "sop_refunds", "text": "Refund process for online and in-store purchases"},
    {"id": "sop_tax_free", "text": "How customers claim tax-free refunds at the airport"},
    {"id": "sop_lost_receipt", "text": "Procedure when a customer cannot produce a receipt"},
    {"id": "sop_damaged_goods", "text": "Steps to handle a customer reporting a damaged purchase"},
]

TEST_CASES = [
    {"input": "How do I claim a tax refund?", "expected": "sop_tax_free"},
    {"input": "What if I lose my receipt?", "expected": "sop_lost_receipt"},
    {"input": "Item arrived broken, what's the process?", "expected": "sop_damaged_goods"},
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
        description="Match a customer question to the most relevant internal SOP.",
        domain_relevance="Front-line staff find the right policy quickly.",
        model=MODEL_ID,
        library="sentence-transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
