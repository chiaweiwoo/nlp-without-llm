"""
Use case: Tag inbound supplier emails by intent / urgency.
Travel-retail relevance: Procurement teams handle dispute, delivery, invoice and routine emails in one inbox.
Model: MoritzLaurer/deberta-v3-base-zeroshot-v2.0
Library: transformers (manual sequence classification NLI evaluation with few-shot context)
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

USE_CASE_ID = "13_fewshot_supplier_urgency"
MODEL_ID = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
TASK_TYPE = "zero_shot_classification"

# 10 labels with their respective descriptive hypotheses for NLI
HYPOTHESES = {
    "dispute": "This email is about overcharges, wrong invoices, or payment discrepancies.",
    "routine": "This email is about monthly catalogues, greetings, or general newsletters.",
    "delivery_update": "This email is about shipment delays, container ETA, or customs holdups.",
    "invoice_query": "This email is about submitting invoices, billing formats, or payment dates.",
    "urgent_escalation": "This email is about a critical stockout, manager escalations, or immediate actions.",
    "price_negotiation": "This email is about wholesale cost, purchase margins, or bulk discounts.",
    "product_specifications": "This email is about material sheets, ingredients, or package dimensions.",
    "contract_renewal": "This email is about contract signoff, SLA reviews, or legal agreements.",
    "out_of_stock_alert": "This email is about vendor production shortages or delayed stock runs.",
    "promotional_campaign": "This email is about marketing plans, seasonal store displays, or co-branding."
}

# Few-shot context containing 1 example for each of the 10 labels
FEW_SHOT_CONTEXT = """Below are examples of supplier email classification:
dispute: "The pricing on the last invoice does not match our contract."
routine: "Best regards, here is our seasonal newsletter."
delivery_update: "Shipment #987 is currently transit-bound and on track."
invoice_query: "Which email address should we send our billing document to?"
urgent_escalation: "EMERGENCY: Store fire has damaged our retail shelves."
price_negotiation: "Can we get a 5% discount if we increase the order to 1000 units?"
product_specifications: "Please find the material safety data sheet attached."
contract_renewal: "Here is the revised procurement SLA contract for review."
out_of_stock_alert: "Our manufacturer is reporting a raw materials delay."
promotional_campaign: "Let's align on the terminal 2 cosmetics promotion layout."

Now, classify this supplier email:
"{text}"
"""

TEST_CASES = [
    {"input": "PO #12345 was overcharged by $500, please refund", "expected": "dispute"},
    {"input": "Please find attached the monthly catalogue update", "expected": "routine"},
    {"input": "Container delayed at customs, ETA pushed two weeks", "expected": "delivery_update"},
    {"input": "We are facing a major inventory shortage, all orders are paused indefinitely", "expected": "out_of_stock_alert"},
    {"input": "CRITICAL: Entire batch of skincare line recalled due to skin irritation report", "expected": "urgent_escalation"},
]

def run() -> dict:
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        
        # Build premise with few-shot context
        premise = FEW_SHOT_CONTEXT.format(text=tc["input"])
        
        scores = {}
        for label, hypothesis in HYPOTHESES.items():
            inputs = tokenizer(premise, hypothesis, return_tensors="pt")
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
        description="Tag inbound supplier emails using custom hypotheses and a 10-class few-shot context.",
        domain_relevance="Procurement teams handle dispute, delivery, invoice and routine emails in one inbox.",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    sys.stdout.buffer.write(json.dumps(run(), indent=2, ensure_ascii=False).encode('utf-8'))
    print()
