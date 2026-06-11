"""
Use case: Categorise customer feedback by topic.
Travel-retail relevance: Travel-retail customer experience teams triage thousands of post-purchase comments.
Model: MoritzLaurer/deberta-v3-base-zeroshot-v2.0
Library: transformers (manual sequence classification NLI evaluation)
"""

import json
import sys
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result

USE_CASE_ID = "09_classify_feedback_topic"
MODEL_ID = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
TASK_TYPE = "zero_shot_classification"

# 10 labels with their respective descriptive hypotheses for NLI
HYPOTHESES = {
    "product_quality": "This feedback is about a damaged product, broken seal, or quality issue.",
    "staff_service": "This feedback is about staff behavior, cashier politeness, or helpfulness.",
    "pricing": "This feedback is about high prices, tax-free value, or exchange rates.",
    "store_experience": "This feedback is about store navigation, layout, or ambient music.",
    "checkout_process": "This feedback is about checkout queues, payment machines, or card errors.",
    "stock_availability": "This feedback is about out of stock items or missing SKU sizes.",
    "tax_free_refund": "This feedback is about claiming tax refunds or customs stamps at the airport.",
    "loyalty_program": "This feedback is about points redemption, membership cards, or signup.",
    "store_cleanliness": "This feedback is about floor hygiene, dust, or trash in the shop.",
    "website_preorder": "This feedback is about online reservation, pickup gate errors, or pre-order site."
}

TEST_CASES = [
    {"input": "The bottle of whisky I bought had a damaged seal", "expected": "product_quality"},
    {"input": "Cashier was rude when I asked for the tax-free form", "expected": "staff_service"},
    {"input": "Too expensive compared to my local market", "expected": "pricing"},
    {"input": "I tried to register my loyalty card but the barcode scanner failed", "expected": "loyalty_program"},
    {"input": "The online pre-order website was down, had to buy at the counter", "expected": "website_preorder"},
]

def run() -> dict:
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        
        # Calculate entailment score for each label's custom hypothesis
        scores = {}
        for label, hypothesis in HYPOTHESES.items():
            inputs = tokenizer(tc["input"], hypothesis, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            # MoritzLaurer/deberta-v3-base-zeroshot-v2.0 uses binary classification (entailment=0, not_entailment=1)
            entail_score = torch.softmax(outputs.logits, dim=-1)[0][0].item()
            scores[label] = entail_score
            
        actual = max(scores, key=scores.get)
        passed = (actual == tc["expected"])
        
        scores_summary = ", ".join(f"{lbl}:{round(score, 3)}" for lbl, score in sorted(scores.items(), key=lambda x: x[1], reverse=True))
        
        results.append({
            "input": tc["input"],
            "expected": tc["expected"],
            "actual": actual,
            "passed": passed,
            "inference_time_s": round(time.perf_counter() - t1, 4),
            "notes": f"scores=[{scores_summary}]",
        })

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Categorise customer feedback by topic using class-specific hypotheses.",
        domain_relevance="Travel-retail customer experience teams triage thousands of post-purchase comments.",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    sys.stdout.buffer.write(json.dumps(run(), indent=2, ensure_ascii=False).encode('utf-8'))
    print()
