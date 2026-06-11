"""
Study runner for nlp-without-llm.

This workflow treats each use case as an independent local study artifact:
- reuse an existing per-use-case JSON result when available
- migrate legacy result files from older numbering when possible
- run only missing baseline scenarios with moderate parallelism
- record known impractical few-shot scenarios without executing them
- rebuild output/results.json and output/report.html from per-use-case files
"""

import argparse
import ast
import glob
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import lab.report
from lab.contract import build_aggregated_results, build_result, save_result

OUTPUT_DIR = Path("output")
RESULTS_JSON_PATH = OUTPUT_DIR / "results.json"
REPORT_HTML_PATH = OUTPUT_DIR / "report.html"
USECASE_GLOB = "usecases/[0-9][0-9]_*.py"
DEFAULT_TIMEOUT_S = 600

KNOWN_LIMITS = {
    "12_fewshot_email_department": (
        "Known local limit: this few-shot NLI prompt expansion took more than 20 minutes "
        "on CPU in prior runs, so the study records it as not practical locally. A hosted "
        "LLM API would likely be a better fit for this scenario."
    ),
    "13_fewshot_supplier_urgency": (
        "Known local limit: this few-shot NLI prompt expansion took more than 20 minutes "
        "on CPU in prior runs, so the study records it as not practical locally. A hosted "
        "LLM API would likely be a better fit for this scenario."
    ),
    "14_fewshot_feedback_topic": (
        "Known local limit: this few-shot NLI prompt expansion took more than 20 minutes "
        "on CPU in prior runs, so the study records it as not practical locally. A hosted "
        "LLM API would likely be a better fit for this scenario."
    ),
}

LEGACY_USE_CASE_IDS = {
    "01_language_detection": "10_language_detection",
    "02_translate_zh_en": "11_translate_zh_en",
    "03_sentiment_customer_review": "04_sentiment_customer_review",
    "04_sentiment_internal_escalation": "05_sentiment_internal_escalation",
    "05_ner_brands_locations": "06_ner_brands_locations",
    "06_ner_people_orgs_procurement": "07_ner_people_orgs_procurement",
    "07_classify_email_department": "01_classify_email_department",
    "08_classify_supplier_urgency": "03_classify_supplier_urgency",
    "09_classify_feedback_topic": "02_classify_feedback_topic",
    "10_similarity_complaint_lookup": "08_similarity_complaint_lookup",
    "11_similarity_sop_retrieval": "09_similarity_sop_retrieval",
    "12_fewshot_email_department": "12_fewshot_email_department",
    "13_fewshot_supplier_urgency": "14_fewshot_supplier_urgency",
    "14_fewshot_feedback_topic": "13_fewshot_feedback_topic",
}


def discover_scripts() -> list[Path]:
    return [Path(path) for path in sorted(glob.glob(USECASE_GLOB))]


def get_script_metadata(script_path: Path) -> dict:
    metadata = {
        "USE_CASE_ID": script_path.stem,
        "MODEL_ID": "unknown",
        "TASK_TYPE": "unknown",
        "TEST_CASES": [],
        "description": f"Use case {script_path.stem}",
    }
    try:
        content = script_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(script_path))
        docstring = ast.get_docstring(tree)
        if docstring:
            lines = [line.strip() for line in docstring.strip().splitlines() if line.strip()]
            if lines:
                metadata["description"] = lines[0]

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name in metadata:
                            try:
                                metadata[name] = ast.literal_eval(node.value)
                            except Exception:
                                pass
    except Exception:
        pass
    return metadata


def get_total_memory_gb() -> float:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024**3)
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3)
    except Exception:
        return 8.0


def get_default_workers() -> int:
    ram_gb = get_total_memory_gb()
    if ram_gb < 10.0:
        return 1
    return 2


def load_result(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)
    if "status" not in result:
        result["status"] = "ok" if result.get("error") is None else "failed"
    return result


def is_reusable_result(result: dict) -> bool:
    return result.get("status") in {"ok", "timeout", "known_limit"}


def build_failed_result(meta: dict, error_msg: str, status: str) -> dict:
    test_cases = meta.get("TEST_CASES", [])
    results = []
    for tc in test_cases:
        results.append(
            {
                "input": tc.get("input", ""),
                "expected": tc.get("expected", ""),
                "actual": None,
                "passed": False,
                "inference_time_s": 0.0,
                "notes": error_msg,
            }
        )

    if not results:
        results.append(
            {
                "input": "N/A",
                "expected": "N/A",
                "actual": None,
                "passed": False,
                "inference_time_s": 0.0,
                "notes": error_msg,
            }
        )

    library = "transformers"
    if meta.get("TASK_TYPE") == "semantic_similarity":
        library = "sentence-transformers"

    return build_result(
        use_case_id=meta["USE_CASE_ID"],
        type=meta["TASK_TYPE"],
        description=meta["description"],
        domain_relevance="N/A" if status != "ok" else "",
        model=meta["MODEL_ID"],
        library=library,
        model_load_time_s=0.0,
        test_cases=results,
        error=error_msg,
        status=status,
    )


def migrate_legacy_result(use_case_id: str) -> dict | None:
    legacy_use_case_id = LEGACY_USE_CASE_IDS.get(use_case_id)
    if not legacy_use_case_id:
        return None

    legacy_path = OUTPUT_DIR / f"result_{legacy_use_case_id}.json"
    if not legacy_path.exists():
        return None

    result = load_result(legacy_path)
    result["use_case_id"] = use_case_id
    if is_reusable_result(result):
        save_result(result, OUTPUT_DIR)
        return result
    return None


def execute_script(script_path: Path, timeout_s: int) -> dict:
    meta = get_script_metadata(script_path)
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return build_failed_result(
            meta,
            (
                f"Timed out after {timeout_s} seconds. Local offline execution was not "
                "practical within the study limit. A hosted LLM API would likely be a "
                "better fit for this scenario."
            ),
            status="timeout",
        )

    if completed.returncode != 0:
        return build_failed_result(
            meta,
            f"Subprocess exited with code {completed.returncode}. Stderr: {completed.stderr[:1000]}",
            status="failed",
        )

    try:
        result = json.loads(completed.stdout.strip())
    except Exception as exc:
        return build_failed_result(
            meta,
            f"Failed to parse JSON from stdout. Parse error: {exc}. Raw stdout: {completed.stdout[:500]}",
            status="failed",
        )

    if "status" not in result:
        result["status"] = "ok" if result.get("error") is None else "failed"
    return result


def write_known_limit_result(script_path: Path) -> dict:
    meta = get_script_metadata(script_path)
    result = build_failed_result(
        meta,
        KNOWN_LIMITS[meta["USE_CASE_ID"]],
        status="known_limit",
    )
    save_result(result, OUTPUT_DIR)
    return result


def ensure_use_case_result(script_path: Path, timeout_s: int) -> tuple[str, dict | None]:
    use_case_id = script_path.stem
    result_path = OUTPUT_DIR / f"result_{use_case_id}.json"

    if use_case_id in KNOWN_LIMITS:
        return "known_limit", write_known_limit_result(script_path)

    if result_path.exists():
        result = load_result(result_path)
        if is_reusable_result(result):
            return "reused", result
        return "pending", None

    migrated = migrate_legacy_result(use_case_id)
    if migrated is not None:
        return "migrated", migrated

    return "pending", None


def aggregate_results_for_scripts(scripts: list[Path]) -> list[dict]:
    use_cases = []
    for script_path in scripts:
        result_path = OUTPUT_DIR / f"result_{script_path.stem}.json"
        if result_path.exists():
            use_cases.append(load_result(result_path))
        elif script_path.stem in KNOWN_LIMITS:
            use_cases.append(write_known_limit_result(script_path))
        else:
            meta = get_script_metadata(script_path)
            use_cases.append(
                build_failed_result(
                    meta,
                    "No result artifact was available for this use case.",
                    status="failed",
                )
            )
    return use_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or reuse per-use-case study artifacts.")
    parser.add_argument("--workers", type=int, default=None, help="Moderate parallel worker count.")
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help="Per-use-case subprocess timeout in seconds.",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Skip HTML report generation after aggregating results.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    t_start = time.perf_counter()

    scripts = discover_scripts()
    pending_scripts: list[Path] = []

    for script_path in scripts:
        state, _ = ensure_use_case_result(script_path, args.timeout_s)
        if state == "pending":
            pending_scripts.append(script_path)

    workers = max(1, args.workers or get_default_workers())
    if pending_scripts:
        workers = min(workers, len(pending_scripts))

    if pending_scripts:
        if workers == 1:
            for script_path in pending_scripts:
                result = execute_script(script_path, args.timeout_s)
                save_result(result, OUTPUT_DIR)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(execute_script, script_path, args.timeout_s): script_path
                    for script_path in pending_scripts
                }
                for future in as_completed(future_map):
                    result = future.result()
                    save_result(result, OUTPUT_DIR)

    finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    aggregated = build_aggregated_results(
        use_cases=aggregate_results_for_scripts(scripts),
        started_at=started_at,
        finished_at=finished_at,
        total_wall_time_s=time.perf_counter() - t_start,
        host_os=f"{platform.system()}-{platform.release()}",
        python_version=platform.python_version(),
    )

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    if not args.skip_html:
        lab.report.render(aggregated, REPORT_HTML_PATH)


if __name__ == "__main__":
    main()
