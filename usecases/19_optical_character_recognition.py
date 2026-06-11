"""
Formal problem: Optical Character Recognition.
Luxury-retail application: Read compact receipt and pickup text from generated images.
Model: microsoft/trocr-small-printed
Method: Transformer OCR
"""

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lab.contract import build_result
from lab.study_utils import load_json_fixture, normalize_text

USE_CASE_ID = "19_optical_character_recognition"
MODEL_ID = "microsoft/trocr-small-printed"
TASK_TYPE = "optical_character_recognition"
PROBLEM_NAME = "Optical Character Recognition"
TECHNIQUE_NAME = "Transformer OCR"
APPLICATION_NAME = "Read compact luxury-retail receipt and pickup text from generated images."
COMPARISON_GROUP = "optical_character_recognition"
RUNTIME_TIER = "heavy"

TEST_CASES = load_json_fixture("optical_character_recognition/examples.json")
IMAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "optical_character_recognition"


def run() -> dict:
    t0 = time.perf_counter()
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    load_time = time.perf_counter() - t0

    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        image = Image.open(IMAGE_DIR / tc["image"]).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        actual = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        passed = normalize_text(actual) == normalize_text(tc["expected"])
        results.append(
            {
                "input": tc["image"],
                "expected": tc["expected"],
                "actual": actual,
                "passed": passed,
                "inference_time_s": round(time.perf_counter() - t1, 4),
                "notes": f"image_size={image.size[0]}x{image.size[1]}",
            }
        )

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="Read compact receipt and pickup text from generated luxury-retail image fixtures.",
        domain_relevance="Receipt-like images are a realistic extension of offline retail workflows beyond plain text.",
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
