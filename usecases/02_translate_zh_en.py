"""
Use case: Translate Chinese customer messages to English for triage.
Travel-retail relevance: Significant traveller segments write feedback in their native language.
Model: Helsinki-NLP/opus-mt-zh-en
Library: transformers (translation) + sentence-transformers (similarity evaluation)
"""

import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from lab.contract import build_result

USE_CASE_ID = "02_translate_zh_en"
MODEL_ID = "facebook/nllb-200-distilled-600M"
TASK_TYPE = "translation"

TEST_CASES = [
    {"input": "请问退税柜台在哪里？", "expected": "Where is the tax refund counter?"},
    {"input": "我想退货", "expected": "I want to return the goods"},
    {"input": "这瓶香水多少钱？", "expected": "How much is this bottle of perfume?"},
]

def run() -> dict:
    t0 = time.perf_counter()
    # Load translation model and tokenizer directly (forcing source lang to Simplified Chinese)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="zho_Hans")
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    # Load sentence similarity model (for evaluation)
    similarity_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        # Perform translation
        inputs = tokenizer(tc["input"], return_tensors="pt")
        translated_tokens = translator_model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=128
        )
        actual = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].strip()
        
        # Calculate cosine similarity with reference translation
        emb1 = similarity_model.encode(actual, convert_to_tensor=True)
        emb2 = similarity_model.encode(tc["expected"], convert_to_tensor=True)
        cosine = round(util.cos_sim(emb1, emb2).item(), 4)
        
        passed = (cosine >= 0.65)
        
        results.append({
            "input": tc["input"],
            "expected": tc["expected"],
            "actual": actual,
            "passed": passed,
            "inference_time_s": round(time.perf_counter() - t1, 4),
            "notes": f"actual='{actual}' cosine={cosine}",
        })

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Translate Chinese customer messages to English for triage.",
        domain_relevance="Significant traveller segments write feedback in their native language.",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    sys.stdout.buffer.write(json.dumps(run(), indent=2, ensure_ascii=False).encode('utf-8'))
    print()
