import glob
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import run_study


def test_use_cases_discovery():
    scripts = glob.glob("usecases/[0-9][0-9]_*.py")
    assert len(scripts) == 14, f"Expected 14 use case scripts, found {len(scripts)}"


def test_build_failed_result_uses_metadata():
    script_path = Path("usecases/01_language_detection.py")
    meta = run_study.get_script_metadata(script_path)

    result = run_study.build_failed_result(meta, "timed out", status="timeout")

    assert result["use_case_id"] == "01_language_detection"
    assert result["status"] == "timeout"
    assert result["error"] == "timed out"
    assert result["pass_count"] == 0
    assert result["total_count"] == len(meta["TEST_CASES"])


@patch("subprocess.run")
def test_runner_records_subprocess_failure(mock_run, tmp_path):
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "Injected subprocess stderr traceback"
    mock_proc.stdout = ""
    mock_run.return_value = mock_proc

    script_path = Path("usecases/03_sentiment_customer_review.py")
    result = run_study.execute_script(script_path, timeout_s=600)

    assert result["use_case_id"] == "03_sentiment_customer_review"
    assert result["status"] == "failed"
    assert "Injected subprocess stderr traceback" in result["error"]

    saved_path = run_study.save_result(result, tmp_path)
    with open(saved_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["status"] == "failed"


@patch("subprocess.run")
def test_runner_handles_malformed_json(mock_run):
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "Malformed raw non-JSON output"
    mock_proc.stderr = ""
    mock_run.return_value = mock_proc

    result = run_study.execute_script(
        Path("usecases/03_sentiment_customer_review.py"), timeout_s=600
    )

    assert result["status"] == "failed"
    assert "Failed to parse JSON" in result["error"]


def test_migrate_legacy_result(tmp_path):
    legacy_result = {
        "use_case_id": "10_language_detection",
        "type": "language_detection",
        "description": "desc",
        "domain_relevance": "relevance",
        "model": "model",
        "library": "transformers",
        "model_load_time_s": 1.0,
        "test_cases": [],
        "pass_count": 0,
        "total_count": 0,
        "pass_rate": 0.0,
        "total_inference_time_s": 0.0,
        "total_runtime_s": 1.0,
        "error": None,
        "status": "ok",
    }
    legacy_path = tmp_path / "result_10_language_detection.json"
    legacy_path.write_text(json.dumps(legacy_result), encoding="utf-8")

    with patch.object(run_study, "OUTPUT_DIR", tmp_path):
        migrated = run_study.migrate_legacy_result("01_language_detection")

    assert migrated is not None
    assert migrated["use_case_id"] == "01_language_detection"
    assert (tmp_path / "result_01_language_detection.json").exists()
