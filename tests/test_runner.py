import glob
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import run_study


def test_use_cases_discovery():
    scripts = glob.glob("usecases/[0-9][0-9]_*.py")
    assert len(scripts) >= 14, f"Expected at least 14 use case scripts, found {len(scripts)}"


def test_build_failed_result_uses_metadata():
    script_path = Path("usecases/01_language_detection.py")
    meta = run_study.get_script_metadata(script_path)

    result = run_study.build_failed_result(meta, "timed out", status="timeout")

    assert result["use_case_id"] == "01_language_detection"
    assert result["status"] == "timeout"
    assert result["error"] == "timed out"
    assert result["pass_count"] == 0
    assert result["total_count"] == len(meta["TEST_CASES"])
    assert result["problem_name"] == "Language Identification"
    assert result["runtime_tier"] == "fast"


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
    assert result["primary_metric"] == "pass_rate"

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
    assert migrated["problem_name"] == "language_detection"
    assert (tmp_path / "result_01_language_detection.json").exists()


def test_filter_scripts_by_only_and_tier():
    scripts = run_study.discover_scripts()

    selected = run_study.filter_scripts(scripts, only="03", max_tier="heavy")
    assert [script.stem for script in selected] == ["03_sentiment_customer_review"]

    selected_all = run_study.filter_scripts(scripts, max_tier="heavy")
    assert len(selected_all) == len(scripts)

    fast_selected = run_study.filter_scripts(scripts, max_tier="fast")
    fast_ids = [script.stem for script in fast_selected]
    assert "01_language_detection" in fast_ids
    assert "03_sentiment_customer_review" in fast_ids
    assert "05_ner_brands_locations" in fast_ids
    assert all(
        run_study.get_script_metadata(script).get("RUNTIME_TIER") == "fast"
        for script in fast_selected
    )


def test_no_arg_main_lists_status(capsys):
    with patch("sys.argv", ["run_study.py"]):
        run_study.main()

    captured = capsys.readouterr()
    assert "Status" in captured.out
    assert "01" in captured.out


def test_report_only_writes_aggregate(tmp_path):
    sample_result = {
        "use_case_id": "01_language_detection",
        "type": "language_detection",
        "description": "desc",
        "domain_relevance": "relevance",
        "model": "model",
        "library": "transformers",
        "model_load_time_s": 0.1,
        "test_cases": [],
        "pass_count": 0,
        "total_count": 0,
        "pass_rate": 0.0,
        "total_inference_time_s": 0.0,
        "total_runtime_s": 0.1,
        "error": None,
        "status": "ok",
    }
    (tmp_path / "result_01_language_detection.json").write_text(
        json.dumps(sample_result), encoding="utf-8"
    )

    with patch.object(run_study, "OUTPUT_DIR", tmp_path), patch.object(
        run_study, "RESULTS_JSON_PATH", tmp_path / "results.json"
    ), patch.object(
        run_study, "REPORT_HTML_PATH", tmp_path / "report.html"
    ), patch(
        "sys.argv", ["run_study.py", "--report-only", "--skip-html", "--only", "01"]
    ):
        run_study.main()

    assert (tmp_path / "results.json").exists()
