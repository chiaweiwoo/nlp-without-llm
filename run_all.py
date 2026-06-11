"""
Runner script for nlp-without-llm.
Discovers and runs all use cases sequentially in subprocesses, aggregates metrics,
saves output/results.json, and renders output/report.html.
"""

import argparse
import glob
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from lab.contract import build_result
import lab.report

def main():
    parser = argparse.ArgumentParser(description="Run all offline NLP use cases.")
    parser.add_argument("--only", type=str, help="Only run a specific use case matching this pattern (e.g., '04')")
    parser.add_argument("--skip-html", action="store_true", help="Skip HTML report generation")
    args = parser.parse_args()

    # Discover scripts in usecases/
    scripts = sorted(glob.glob("usecases/[0-9][0-9]_*.py"))
    
    if args.only:
        scripts = [s for s in scripts if args.only in Path(s).name]
        if not scripts:
            print(f"No use cases matched pattern: {args.only}")
            sys.exit(1)

    print(f"Discovered {len(scripts)} use cases to run.")
    
    # Ensure output directory exists
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    use_cases_results = []
    
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    t_start = time.perf_counter()

    for script_path in scripts:
        script_path = Path(script_path)
        print(f"Running {script_path.name}...")
        
        # Capture wall time for this run
        t_run_start = time.perf_counter()
        
        # Run script in a subprocess using the current python interpreter
        # Use errors="replace" and encoding="utf-8" to handle any console encoding issues
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        
        run_wall_time = time.perf_counter() - t_run_start

        if result.returncode == 0:
            try:
                # Parse stdout as JSON
                result_json = json.loads(result.stdout.strip())
                # Update total runtime if not set or just verify
                use_cases_results.append(result_json)
                print(f"  Passed: {result_json['pass_count']}/{result_json['total_count']} test cases. (Runtime: {round(run_wall_time, 2)}s)")
            except Exception as e:
                # Failed to parse stdout as JSON
                error_msg = f"Failed to parse JSON from stdout. Parse error: {str(e)}. Raw stdout: {result.stdout[:500]}"
                use_cases_results.append(handle_failure(script_path, error_msg))
                print(f"  Error: {error_msg}")
        else:
            # Subprocess exited with non-zero code
            error_msg = f"Subprocess exited with code {result.returncode}. Stderr: {result.stderr[:1000]}"
            use_cases_results.append(handle_failure(script_path, error_msg))
            print(f"  Error: {error_msg}")

    total_wall_time = time.perf_counter() - t_start
    finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Compute summaries
    total_use_cases = len(use_cases_results)
    successful_use_cases = sum(1 for uc in use_cases_results if uc.get("error") is None)
    total_test_cases = sum(len(uc.get("test_cases", [])) for uc in use_cases_results)
    total_passed = sum(sum(1 for tc in uc.get("test_cases", []) if tc.get("passed", False)) for uc in use_cases_results)
    overall_pass_rate = round(total_passed / total_test_cases, 4) if total_test_cases > 0 else 0.0

    aggregated_results = {
        "run_metadata": {
            "started_at": started_at,
            "finished_at": finished_at,
            "total_wall_time_s": round(total_wall_time, 4),
            "host_os": f"{platform.system()}-{platform.release()}",
            "python_version": platform.python_version()
        },
        "use_cases": use_cases_results,
        "summary": {
            "total_use_cases": total_use_cases,
            "successful_use_cases": successful_use_cases,
            "total_test_cases": total_test_cases,
            "total_passed": total_passed,
            "overall_pass_rate": overall_pass_rate
        }
    }

    # Write results.json
    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_results, f, indent=2, ensure_ascii=False)
    print(f"Results written to {results_json_path}")

    # Generate HTML report
    if not args.skip_html:
        report_path = output_dir / "report.html"
        lab.report.render(aggregated_results, report_path)

def handle_failure(script_path: Path, error_msg: str) -> dict:
    """Helper to build a failure result schema when a script fails to execute."""
    try:
        # Dynamic import to attempt to fetch module-level variables
        spec = importlib.util.spec_from_file_location("temp_module", str(script_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        test_cases_defs = getattr(module, "TEST_CASES", [])
        use_case_id = getattr(module, "USE_CASE_ID", script_path.stem)
        model_id = getattr(module, "MODEL_ID", "unknown")
        task_type = getattr(module, "TASK_TYPE", "unknown")
    except Exception:
        test_cases_defs = []
        use_case_id = script_path.stem
        model_id = "unknown"
        task_type = "unknown"

    test_cases_results = []
    for tc in test_cases_defs:
        test_cases_results.append({
            "input": tc.get("input", ""),
            "expected": tc.get("expected", ""),
            "actual": None,
            "passed": False,
            "inference_time_s": 0.0,
            "notes": f"Execution failed: {error_msg}"
        })

    # Fallback if no test cases were defined or readable
    if not test_cases_results:
        test_cases_results.append({
            "input": "N/A",
            "expected": "N/A",
            "actual": None,
            "passed": False,
            "inference_time_s": 0.0,
            "notes": f"Execution failed: {error_msg}"
        })

    return build_result(
        use_case_id=use_case_id,
        type=task_type,
        description=f"Failed use case {use_case_id}",
        domain_relevance="N/A",
        model=model_id,
        library="unknown",
        model_load_time_s=0.0,
        test_cases=test_cases_results,
        error=error_msg
    )

if __name__ == "__main__":
    main()
