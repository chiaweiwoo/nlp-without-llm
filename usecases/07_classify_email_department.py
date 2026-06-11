"""
Use case: Route internal emails to the correct department.
Travel-retail relevance: Multi-department retail operations get hundreds of emails; auto-routing cuts triage time.
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

USE_CASE_ID = "07_classify_email_department"
MODEL_ID = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
TASK_TYPE = "zero_shot_classification"

# 10 labels with their respective descriptive hypotheses for NLI
HYPOTHESES = {
    "IT": "This email is about technical support, server down, or software issues.",
    "Planner": "This email is about inventory planning, stock replenishment, or purchasing.",
    "Pricing": "This email is about duty-free pricing, retail margins, or discounts.",
    "Warehouse": "This email is about logistics, shipping containers, or physical storage.",
    "Finance": "This email is about invoicing, accounting, or budget approval.",
    "HR": "This email is about job recruitment, employee payroll, or training.",
    "CustomerService": "This email is about customer feedback, complaints, or return requests.",
    "Procurement": "This email is about supplier contracts, vendor agreements, or wholesale terms.",
    "Marketing": "This email is about airport advertisements, loyalty programs, or retail campaigns.",
    "Security": "This email is about store security, loss prevention, or customs compliance."
}

TEST_CASES = [
    {"input": "Server is down, checkouts failing", "expected": "IT"},
    {"input": "Need to reorder fragrance stock for the airport store, running low", "expected": "Planner"},
    {"input": "Q3 expense report due Friday", "expected": "Finance"},
    {"input": "Requesting supplier contract renewal terms for next year", "expected": "Procurement"},
    {"input": "A customer reported a stolen handbag at the gate 3 shop", "expected": "Security"},
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
        
        # Sort scores summary for verbose logging in report notes
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
        description="Route incoming internal emails to the correct department using class-specific hypotheses.",
        domain_relevance="Reduces manual triage in a multi-department retail org.",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    sys.stdout.buffer.write(json.dumps(run(), indent=2, ensure_ascii=False).encode('utf-8'))
    print()
