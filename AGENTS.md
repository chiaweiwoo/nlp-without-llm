# Repository Agent Guide

## Project Overview

This repository demonstrates offline pretrained NLP models for travel-retail
use cases. It uses Python 3.11+, `uv`, Hugging Face Transformers, PyTorch, and
Sentence Transformers.

The main entry points are:

- `run_study.py`: discovers use-case scripts, reuses or creates per-use-case
  result artifacts, writes `output/results.json`, and optionally renders
  `output/report.html`.
- `usecases/NN_*.py`: standalone model-backed examples that print one JSON
  result to stdout.
- `lab/contract.py`: defines the shared result schema via `build_result`.
- `lab/report.py`: renders the aggregated HTML report.
- `tests/`: fast unit tests for runner and report behavior.

## Setup And Commands

Use `uv` for dependency and command execution:

```bash
uv sync
uv run python -m pytest -q
uv run python usecases/04_sentiment_internal_escalation.py
uv run python run_study.py --skip-html
uv run python run_study.py
```

Run the unit tests before model-backed evaluations. A full evaluation can
download several gigabytes of model weights, consume substantial memory, and
take several minutes. Prefer running a single use-case script while developing
that scenario.

## Implementation Conventions

- Keep each use case independently executable.
- Name use-case files `NN_descriptive_name.py` with a two-digit prefix.
- Define `USE_CASE_ID`, `MODEL_ID`, `TASK_TYPE`, and `TEST_CASES` at module
  level. `USE_CASE_ID` should match the filename stem.
- Implement `run() -> dict` and return `lab.contract.build_result(...)`.
- Keep stdout machine-readable: the script's successful stdout must be a
  single valid JSON document. Send diagnostics elsewhere or include them in
  result fields.
- Record model load time separately from per-case inference time.
- Every test-case result should include `input`, `expected`, `actual`,
  `passed`, `inference_time_s`, and useful `notes`.
- Preserve Unicode by serializing with `ensure_ascii=False` and UTF-8.
- Do not add network APIs or hosted LLM dependencies. Inference is intended to
  work offline after model weights are cached.
- Keep report generation driven by the aggregated result schema rather than
  hard-coding individual use cases.

## Testing Expectations

- Add or update focused unit tests for runner, contract, or report changes.
- Mock subprocesses and model boundaries in unit tests; unit tests must not
  download or load large models.
- When adding or removing a use-case script, update discovery-count assertions
  and user-facing documentation.
- For a use-case change, run the unit suite first, then run only the affected
  use case when cached model weights and resources are available.
- Do not commit generated `output/results.json`, `output/report.html`, caches,
  virtual environments, or model weights.

## Change Discipline

- Keep changes scoped to the requested behavior.
- Preserve existing user work in a dirty worktree.
- Avoid unrelated formatting or dependency-lock churn.
- Update `README.md` when use-case counts, commands, models, or reported
  capabilities change.
