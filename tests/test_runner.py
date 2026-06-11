import json
import glob
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import run_all

def test_use_cases_discovery():
    """Verify that all 11 use case files are present and match the expected naming pattern."""
    scripts = glob.glob("usecases/[0-9][0-9]_*.py")
    assert len(scripts) == 11, f"Expected 11 use case scripts, found {len(scripts)}"

def test_handle_failure_helper():
    """Test that run_all.handle_failure returns a schema-compliant error dictionary."""
    dummy_path = Path("usecases/99_dummy_nonexistent.py")
    error_msg = "Test injected error message"
    result = run_all.handle_failure(dummy_path, error_msg)
    
    assert result["use_case_id"] == "99_dummy_nonexistent"
    assert result["error"] == error_msg
    assert result["pass_count"] == 0
    assert result["total_count"] == 1
    assert result["pass_rate"] == 0.0
    assert len(result["test_cases"]) == 1
    assert result["test_cases"][0]["passed"] is False
    assert error_msg in result["test_cases"][0]["notes"]

@patch("subprocess.run")
def test_runner_graceful_subprocess_failure(mock_run, tmp_path):
    """Test that the runner gracefully records an error and doesn't crash when a script returns exit code 1."""
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "Injected subprocess stderr traceback"
    mock_proc.stdout = ""
    mock_run.return_value = mock_proc

    with patch("sys.argv", ["run_all.py", "--only", "04", "--skip-html"]):
        original_open = open
        def mock_open(file, mode="r", *args, **kwargs):
            if "results.json" in str(file):
                return original_open(tmp_path / "results.json", mode, *args, **kwargs)
            return original_open(file, mode, *args, **kwargs)

        with patch("run_all.open", mock_open):
            run_all.main()

        results_json = tmp_path / "results.json"
        assert results_json.exists()
        with open(results_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Verify results.json keys and schema types
        assert "run_metadata" in data
        assert "use_cases" in data
        assert "summary" in data
        assert data["summary"]["total_use_cases"] == 1
        assert data["summary"]["successful_use_cases"] == 0
        assert len(data["use_cases"]) == 1
        assert data["use_cases"][0]["use_case_id"] == "04_sentiment_customer_review"
        assert data["use_cases"][0]["error"] is not None
        assert "Injected subprocess stderr traceback" in data["use_cases"][0]["error"]

@patch("subprocess.run")
def test_runner_malformed_json_handling(mock_run, tmp_path):
    """Test that the runner gracefully handles malformed JSON output from use case scripts."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "Malformed raw non-JSON output"
    mock_run.return_value = mock_proc

    with patch("sys.argv", ["run_all.py", "--only", "04", "--skip-html"]):
        original_open = open
        def mock_open(file, mode="r", *args, **kwargs):
            if "results.json" in str(file):
                return original_open(tmp_path / "results.json", mode, *args, **kwargs)
            return original_open(file, mode, *args, **kwargs)

        with patch("run_all.open", mock_open):
            run_all.main()

        results_json = tmp_path / "results.json"
        assert results_json.exists()
        with open(results_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["summary"]["successful_use_cases"] == 0
        assert len(data["use_cases"]) == 1
        assert "Failed to parse JSON" in data["use_cases"][0]["error"]
