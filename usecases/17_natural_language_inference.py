"""
Formal problem: Natural Language Inference.
Luxury-retail application: Verify promotion compliance reports for entailment, neutrality, or contradiction.
Model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
Method: Three-way NLI classification
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import load_json_fixture

USE_CASE_ID = "17_natural_language_inference"
MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
TASK_TYPE = "natural_language_inference"
PROBLEM_NAME = "Natural Language Inference"
TECHNIQUE_NAME = "Three-way NLI classification"
APPLICATION_NAME = "Verify luxury-retail promotion compliance reports for entailment, neutrality, or contradiction."
COMPARISON_GROUP = "natural_language_inference"
RUNTIME_TIER = "medium"

TEST_CASES = load_json_fixture("natural_language_inference/examples.json")


def run() -> dict:
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    id_to_label = {int(idx): label.lower() for idx, label in model.config.id2label.items()}
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        inputs = tokenizer(tc["premise"], tc["hypothesis"], return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        best_idx = int(torch.argmax(probs).item())
        actual = id_to_label.get(best_idx, str(best_idx))
        passed = actual == tc["expected"]
        notes = ", ".join(
            f"{id_to_label[idx]}:{round(probs[idx].item(), 3)}"
            for idx in range(len(probs))
        )
        results.append(
            {
                "input": {
                    "premise": tc["premise"],
                    "hypothesis": tc["hypothesis"],
                },
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"class_probs=[{notes}]",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Classify luxury-retail promotion report pairs as entailment, neutral, or contradiction.",
        domain_relevance="Compliance checks often depend on whether a store report confirms, omits, or contradicts the campaign brief.",
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
