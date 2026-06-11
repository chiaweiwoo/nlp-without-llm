"""
Study runner for nlp-without-llm.

This workflow treats each use case as an independent local study artifact:
- discover standalone study scripts
- list study status without executing by default
- run selected studies on demand
- reuse or refresh per-use-case JSON results
- rebuild output/results.json and output/report.html from saved artifacts
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
from lab.study_utils import load_json_fixture

OUTPUT_DIR = Path("output")
RESULTS_JSON_PATH = OUTPUT_DIR / "results.json"
REPORT_HTML_PATH = OUTPUT_DIR / "report.html"
USECASE_GLOB = "usecases/[0-9][0-9]_*.py"
DEFAULT_TIMEOUT_S = 600
RUNTIME_TIER_ORDER = {"fast": 0, "medium": 1, "heavy": 2}

KNOWN_LIMITS = {}

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
        "PROBLEM_NAME": "unknown",
        "TECHNIQUE_NAME": "unknown",
        "APPLICATION_NAME": "",
        "COMPARISON_GROUP": "",
        "RUNTIME_TIER": "medium",
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
                                if (
                                    name == "TEST_CASES"
                                    and isinstance(node.value, ast.Call)
                                    and isinstance(node.value.func, ast.Name)
                                    and node.value.func.id == "load_json_fixture"
                                    and node.value.args
                                    and isinstance(node.value.args[0], ast.Constant)
                                    and isinstance(node.value.args[0].value, str)
                                ):
                                    metadata[name] = load_json_fixture(node.value.args[0].value)
    except Exception:
        pass

    if metadata["PROBLEM_NAME"] == "unknown":
        metadata["PROBLEM_NAME"] = metadata["TASK_TYPE"]
    if metadata["TECHNIQUE_NAME"] == "unknown":
        metadata["TECHNIQUE_NAME"] = metadata["MODEL_ID"]
    if not metadata["APPLICATION_NAME"]:
        metadata["APPLICATION_NAME"] = metadata["description"]
    if not metadata["COMPARISON_GROUP"]:
        metadata["COMPARISON_GROUP"] = metadata["PROBLEM_NAME"]
    if metadata["RUNTIME_TIER"] not in RUNTIME_TIER_ORDER:
        metadata["RUNTIME_TIER"] = "medium"
    return metadata


def normalize_result(result: dict) -> dict:
    if "status" not in result:
        result["status"] = "ok" if result.get("error") is None else "failed"
    result.setdefault("problem_name", result.get("type", "unknown"))
    result.setdefault("technique_name", result.get("library", "unknown"))
    result.setdefault("application_name", result.get("description", ""))
    result.setdefault("comparison_group", result.get("problem_name", result.get("type", "unknown")))
    result.setdefault("runtime_tier", "medium")
    result.setdefault("primary_metric", "pass_rate")
    result.setdefault("metrics", {"pass_rate": result.get("pass_rate", 0.0)})
    result.setdefault("completed_at", None)
    return result


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
    return normalize_result(result)


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
        domain_relevance="N/A",
        model=meta["MODEL_ID"],
        library=library,
        model_load_time_s=0.0,
        test_cases=results,
        error=error_msg,
        status=status,
        problem_name=meta.get("PROBLEM_NAME"),
        technique_name=meta.get("TECHNIQUE_NAME"),
        application_name=meta.get("APPLICATION_NAME"),
        comparison_group=meta.get("COMPARISON_GROUP"),
        runtime_tier=meta.get("RUNTIME_TIER", "medium"),
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

    return normalize_result(result)


def write_known_limit_result(script_path: Path) -> dict:
    meta = get_script_metadata(script_path)
    result = build_failed_result(
        meta,
        KNOWN_LIMITS[meta["USE_CASE_ID"]],
        status="known_limit",
    )
    save_result(result, OUTPUT_DIR)
    return result


def ensure_use_case_result(
    script_path: Path,
    timeout_s: int,
    force: bool = False,
) -> tuple[str, dict | None]:
    use_case_id = script_path.stem
    result_path = OUTPUT_DIR / f"result_{use_case_id}.json"

    if not force and use_case_id in KNOWN_LIMITS:
        return "known_limit", write_known_limit_result(script_path)

    if not force and result_path.exists():
        result = load_result(result_path)
        if is_reusable_result(result):
            return "reused", result
        return "pending", None

    if not force:
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


def matches_runtime_tier(meta: dict, max_tier: str) -> bool:
    tier = meta.get("RUNTIME_TIER", "medium")
    return RUNTIME_TIER_ORDER[tier] <= RUNTIME_TIER_ORDER[max_tier]


def matches_problem(meta: dict, selector: str) -> bool:
    normalized_selector = selector.strip().lower().replace(" ", "_")
    candidates = {
        meta.get("PROBLEM_NAME", ""),
        meta.get("TASK_TYPE", ""),
        meta.get("USE_CASE_ID", ""),
    }
    for candidate in candidates:
        normalized_candidate = candidate.strip().lower().replace(" ", "_")
        if normalized_candidate == normalized_selector:
            return True
    return False


def filter_scripts(
    scripts: list[Path],
    only: str | None = None,
    problem: str | None = None,
    max_tier: str = "heavy",
) -> list[Path]:
    selected = []
    for script_path in scripts:
        meta = get_script_metadata(script_path)
        if only and not (
            script_path.stem == only
            or script_path.stem.startswith(f"{only}_")
            or script_path.name.startswith(f"{only}_")
        ):
            continue
        if problem and not matches_problem(meta, problem):
            continue
        if not matches_runtime_tier(meta, max_tier):
            continue
        selected.append(script_path)
    return selected


def print_study_status(scripts: list[Path]) -> None:
    print("ID  Tier    Status        Problem")
    print("--  ------  ------------  -------")
    for script_path in scripts:
        meta = get_script_metadata(script_path)
        result_path = OUTPUT_DIR / f"result_{script_path.stem}.json"
        if result_path.exists():
            status = load_result(result_path).get("status", "unknown")
        elif script_path.stem in KNOWN_LIMITS:
            status = "known_limit"
        else:
            status = "not_run"
        print(
            f"{script_path.stem[:2]}  "
            f"{meta.get('RUNTIME_TIER', 'medium'):<6}  "
            f"{status:<12}  "
            f"{meta.get('PROBLEM_NAME', 'unknown')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or inspect local study artifacts.")
    parser.add_argument("--list", action="store_true", help="List study status without executing.")
    parser.add_argument("--all", action="store_true", help="Run all selected studies.")
    parser.add_argument("--only", help="Run one study by numeric prefix or full use-case id.")
    parser.add_argument("--problem", help="Run all studies for one problem name.")
    parser.add_argument("--report-only", action="store_true", help="Render aggregate outputs from saved results only.")
    parser.add_argument("--force", action="store_true", help="Re-run even when a reusable result exists.")
    parser.add_argument(
        "--max-tier",
        choices=sorted(RUNTIME_TIER_ORDER, key=RUNTIME_TIER_ORDER.get),
        default="heavy",
        help="Maximum runtime tier to include.",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    scripts = filter_scripts(
        discover_scripts(),
        only=args.only,
        problem=args.problem,
        max_tier=args.max_tier,
    )

    if args.report_only:
        started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        aggregated = build_aggregated_results(
            use_cases=aggregate_results_for_scripts(scripts),
            started_at=started_at,
            finished_at=started_at,
            total_wall_time_s=0.0,
            host_os=f"{platform.system()}-{platform.release()}",
            python_version=platform.python_version(),
        )
        with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        if not args.skip_html:
            lab.report.render(aggregated, REPORT_HTML_PATH)
        return

    should_execute = bool(args.all or args.only or args.problem)
    if args.list or not should_execute:
        print_study_status(scripts)
        return

    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    t_start = time.perf_counter()
    pending_scripts: list[Path] = []

    for script_path in scripts:
        state, _ = ensure_use_case_result(script_path, args.timeout_s, force=args.force)
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
