import json
import typing
from pathlib import Path


def build_result(
    use_case_id: str,
    type: str,
    description: str,
    domain_relevance: str,
    model: str,
    library: str,
    model_load_time_s: float,
    test_cases: list[dict],
    error: typing.Optional[str] = None,
    status: str | None = None,
) -> dict:
    """
    Build the standard result dictionary for a use case.
    """

    pass_count = sum(1 for tc in test_cases if tc.get("passed", False))
    total_count = len(test_cases)
    pass_rate = round(pass_count / total_count, 4) if total_count > 0 else 0.0
    total_inference_time_s = round(
        sum(tc.get("inference_time_s", 0.0) for tc in test_cases), 4
    )
    total_runtime_s = round(model_load_time_s + total_inference_time_s, 4)

    resolved_status = status
    if resolved_status is None:
        resolved_status = "ok" if error is None else "failed"

    return {
        "use_case_id": use_case_id,
        "type": type,
        "description": description,
        "domain_relevance": domain_relevance,
        "model": model,
        "library": library,
        "model_load_time_s": model_load_time_s,
        "test_cases": test_cases,
        "pass_count": pass_count,
        "total_count": total_count,
        "pass_rate": pass_rate,
        "total_inference_time_s": total_inference_time_s,
        "total_runtime_s": total_runtime_s,
        "error": error,
        "status": resolved_status,
    }


def save_result(result: dict, output_dir: Path = Path("output")) -> Path:
    output_dir.mkdir(exist_ok=True)
    result_json_path = output_dir / f"result_{result['use_case_id']}.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result_json_path


def build_aggregated_results(
    use_cases: list[dict],
    started_at: str,
    finished_at: str,
    total_wall_time_s: float,
    host_os: str,
    python_version: str,
) -> dict:
    total_use_cases = len(use_cases)
    successful_use_cases = sum(1 for uc in use_cases if uc.get("status") == "ok")
    total_test_cases = sum(len(uc.get("test_cases", [])) for uc in use_cases)
    total_passed = sum(
        sum(1 for tc in uc.get("test_cases", []) if tc.get("passed", False))
        for uc in use_cases
    )
    overall_pass_rate = (
        round(total_passed / total_test_cases, 4) if total_test_cases > 0 else 0.0
    )

    return {
        "run_metadata": {
            "started_at": started_at,
            "finished_at": finished_at,
            "total_wall_time_s": round(total_wall_time_s, 4),
            "host_os": host_os,
            "python_version": python_version,
        },
        "use_cases": use_cases,
        "summary": {
            "total_use_cases": total_use_cases,
            "successful_use_cases": successful_use_cases,
            "total_test_cases": total_test_cases,
            "total_passed": total_passed,
            "overall_pass_rate": overall_pass_rate,
        },
    }
