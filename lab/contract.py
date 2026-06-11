import typing

def build_result(
    use_case_id: str,
    type: str,
    description: str,
    domain_relevance: str,
    model: str,
    library: str,
    model_load_time_s: float,
    test_cases: list[dict],
    error: typing.Optional[str] = None
) -> dict:
    """
    Builds the standard result dictionary for a use case,
    calculating derived metrics such as pass_count, pass_rate, and totals.
    """
    pass_count = sum(1 for tc in test_cases if tc.get("passed", False))
    total_count = len(test_cases)
    pass_rate = round(pass_count / total_count, 4) if total_count > 0 else 0.0
    total_inference_time_s = round(sum(tc.get("inference_time_s", 0.0) for tc in test_cases), 4)
    total_runtime_s = round(model_load_time_s + total_inference_time_s, 4)
    
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
        "error": error
    }
