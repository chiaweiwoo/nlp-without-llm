"""
Formal problem: Machine Translation.
Luxury-retail application: Translate Mandarin luxury-retail enquiries to English for local staff.
Model: facebook/nllb-200-distilled-600M
Method: Encoder-decoder transformer
"""

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from lab.contract import build_result
from lab.study_utils import load_json_fixture

USE_CASE_ID = "02_translate_zh_en"
MODEL_ID = "facebook/nllb-200-distilled-600M"
TASK_TYPE = "translation"
PROBLEM_NAME = "Machine Translation"
TECHNIQUE_NAME = "Encoder-decoder transformer"
APPLICATION_NAME = "Translate Mandarin luxury-retail enquiries into English for boutique and airport staff."
COMPARISON_GROUP = "machine_translation"
RUNTIME_TIER = "heavy"

TEST_CASES = load_json_fixture("machine_translation/examples.json")


def run() -> dict:
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="zho_Hans")
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    similarity_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        inputs = tokenizer(tc["input"], return_tensors="pt")
        translated_tokens = translator_model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=128,
        )
        actual = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].strip()

        emb1 = similarity_model.encode(actual, convert_to_tensor=True)
        emb2 = similarity_model.encode(tc["expected"], convert_to_tensor=True)
        cosine = round(util.cos_sim(emb1, emb2).item(), 4)
        passed = cosine >= 0.65

        results.append(
            {
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"actual='{actual}' cosine={cosine}",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Translate Mandarin luxury-retail enquiries into English for staff triage.",
        domain_relevance="Airport luxury boutiques often need quick local translation before boarding windows close.",
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
    sys.stdout.buffer.write(json.dumps(run(), indent=2, ensure_ascii=False).encode("utf-8"))
    print()
