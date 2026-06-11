import html.parser
from pathlib import Path

import lab.report
import pytest


class SimpleHTMLParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.error_occurred = False

    def handle_error(self, message):
        self.error_occurred = True


def test_report_generation(tmp_path):
    sample_results = {
        "run_metadata": {
            "started_at": "2026-06-11T14:30:00Z",
            "finished_at": "2026-06-11T14:35:00Z",
            "total_wall_time_s": 300.0,
            "host_os": "TestOS",
            "python_version": "3.11.0",
        },
        "use_cases": [
            {
                "use_case_id": "01_language_detection",
                "type": "language_detection",
                "description": "Test description 1",
                "domain_relevance": "Test relevance 1",
                "model": "test-model-1",
                "library": "transformers",
                "model_load_time_s": 1.0,
                "test_cases": [
                    {
                        "input": "in1",
                        "expected": "exp1",
                        "actual": "exp1",
                        "passed": True,
                        "inference_time_s": 0.1,
                        "notes": "n1",
                    }
                ],
                "pass_count": 1,
                "total_count": 1,
                "pass_rate": 1.0,
                "total_inference_time_s": 0.1,
                "total_runtime_s": 1.1,
                "error": None,
                "status": "ok",
                "problem_name": "Language Identification",
                "technique_name": "Transformer classification",
                "application_name": "Identify the language of a traveller enquiry.",
                "comparison_group": "language_identification",
                "runtime_tier": "fast",
                "primary_metric": "pass_rate",
                "metrics": {"pass_rate": 1.0},
                "completed_at": "2026-06-12T10:00:00Z",
            },
            {
                "use_case_id": "03_sentiment_customer_review",
                "type": "sentiment",
                "description": "Test description 2",
                "domain_relevance": "Test relevance 2",
                "model": "test-model-2",
                "library": "transformers",
                "model_load_time_s": 2.0,
                "test_cases": [
                    {
                        "input": "in2",
                        "expected": "exp2",
                        "actual": "fail2",
                        "passed": False,
                        "inference_time_s": 0.2,
                        "notes": "n2",
                    }
                ],
                "pass_count": 0,
                "total_count": 1,
                "pass_rate": 0.0,
                "total_inference_time_s": 0.2,
                "total_runtime_s": 2.2,
                "error": "Dummy error",
                "status": "failed",
                "problem_name": "Sentiment Analysis",
                "technique_name": "Transformer classifier",
                "application_name": "Evaluate luxury-retail review sentiment.",
                "comparison_group": "sentiment_analysis",
                "runtime_tier": "medium",
                "primary_metric": "pass_rate",
                "metrics": {"pass_rate": 0.0},
                "completed_at": "2026-06-12T10:01:00Z",
            },
        ],
        "summary": {
            "total_use_cases": 2,
            "successful_use_cases": 1,
            "total_test_cases": 2,
            "total_passed": 1,
            "overall_pass_rate": 0.5,
        },
    }

    report_file = tmp_path / "report.html"
    lab.report.render(sample_results, report_file)

    assert report_file.exists()
    html_content = report_file.read_text(encoding="utf-8")

    assert "01_language_detection" in html_content
    assert "03_sentiment_customer_review" in html_content
    assert "test-model-1" in html_content
    assert "test-model-2" in html_content
    assert "50.0%" in html_content
    assert "<table>" in html_content

    parser = SimpleHTMLParser()
    try:
        parser.feed(html_content)
    except Exception as exc:
        pytest.fail(f"HTML parser raised an exception: {exc}")
    assert not parser.error_occurred
