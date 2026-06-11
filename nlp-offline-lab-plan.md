# `nlp-offline-lab` — Build Plan

A self-contained build specification for an AI coding agent (Claude Code, Codex, Cursor, Antigravity, etc.). Read this end-to-end before writing any code. Every section is binding unless explicitly marked optional.

---

## 1. Mission

Build a learning repository that demonstrates **offline pretrained NLP models** (no LLM APIs, no internet at inference time) on **travel-retail / duty-free** business scenarios. Each use case is a single Python script. A runner executes all of them, collects metrics, and renders an HTML report.

**Purpose of the repo:** education + portfolio piece. It must run end-to-end on a fresh machine with one command after setup. The reader should be able to open any single script and understand the model + task in isolation.

**Domain framing:** travel retail / duty-free (airport stores, traveller customers, luxury brand suppliers). Do NOT name any specific real company anywhere in code, comments, README, or report. Use generic personas: "the retailer", "the store", "a supplier", etc.

---

## 2. Hard constraints

- **No paid API calls.** No OpenAI, Anthropic, DeepSeek, Gemini, etc. All inference is local via HuggingFace `transformers` and `sentence-transformers`.
- **Internet allowed only on first model download.** All scripts must work offline after that (HF cache handles this automatically).
- **CPU-only.** Do not use CUDA, do not check for GPU, do not branch on device. Models chosen are small enough for CPU inference.
- **Python 3.11+.** Lock via `pyproject.toml`.
- **Dependency manager: `uv`.** Use `uv init`, `uv add`, `uv run`. No `pip` in instructions to the user.
- **No proprietary or trademarked company names** in any artefact (code, data, README, report). Generic personas only.

---

## 3. Repo & git workflow

### Repo name
`nlp-offline-lab`

### Setup commands (run by the agent, in order)
```bash
# 1. Create directory and init
mkdir nlp-offline-lab
cd nlp-offline-lab
git init -b main
uv init --package --name nlp-offline-lab

# 2. Create GitHub remote (public) and push initial scaffold
# Assumes `gh` CLI is installed and authenticated. If `gh` is missing,
# stop and ask the user to install it or provide a remote URL manually.
gh repo create nlp-offline-lab --public --source=. --remote=origin --description "Offline pretrained NLP models for travel retail use cases"

# 3. First commit (after creating .gitignore, README skeleton, pyproject.toml)
git add .
git commit -m "chore: initial scaffold"
git push -u origin main
```

### Commit cadence (mandatory)
Commit **and push** after each of these milestones. One commit per milestone, descriptive message:

1. `chore: initial scaffold` — repo init, gitignore, empty README, pyproject.toml
2. `feat: add shared utils and use case contract` — `lab/contract.py` and any shared helpers
3. `feat: implement use case 04 sentiment customer review` — first script working end-to-end (chosen as the template)
4. `feat: implement use cases 01-03 zero-shot classification`
5. `feat: implement use case 05 sentiment internal escalation`
6. `feat: implement use cases 06-07 named entity recognition`
7. `feat: implement use cases 08-09 semantic similarity`
8. `feat: implement use cases 10-11 multilingual`
9. `feat: add run_all runner`
10. `feat: add HTML report generator`
11. `test: add runner and report unit tests`
12. `docs: finalise README with usage and results summary`
13. `fix: ...` — any iterative fixes from the verification pass

If a commit lumps unrelated changes, split it. If a milestone produces no real change, skip the commit.

### `.gitignore` (minimum contents)
```
__pycache__/
*.pyc
.venv/
.uv/
.idea/
.vscode/
.DS_Store
output/results.json
output/report.html
*.egg-info/
.pytest_cache/
```

---

## 4. Directory structure (final state)

```
nlp-offline-lab/
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
├── lab/
│   ├── __init__.py
│   ├── contract.py             # shared types + result schema helpers
│   └── report.py               # HTML report generator (used by run_all)
├── usecases/
│   ├── __init__.py
│   ├── 01_classify_email_department.py
│   ├── 02_classify_feedback_topic.py
│   ├── 03_classify_supplier_urgency.py
│   ├── 04_sentiment_customer_review.py
│   ├── 05_sentiment_internal_escalation.py
│   ├── 06_ner_brands_locations.py
│   ├── 07_ner_people_orgs_procurement.py
│   ├── 08_similarity_complaint_lookup.py
│   ├── 09_similarity_sop_retrieval.py
│   ├── 10_language_detection.py
│   └── 11_translate_zh_en.py
├── tests/
│   ├── __init__.py
│   ├── test_runner.py
│   └── test_report.py
├── output/                     # gitignored; created at runtime
│   ├── results.json
│   └── report.html
└── run_all.py
```

---

## 5. Dependencies (`pyproject.toml`)

Use `uv add` for each. Final dependency list:

| Package | Purpose |
|---|---|
| `transformers>=4.44` | HF model loading and pipelines |
| `torch>=2.3` (CPU build) | Backend for transformers |
| `sentence-transformers>=3.0` | Embeddings (use cases 08, 09, 11 eval) |
| `sentencepiece` | Required by some MT/multilingual tokenizers |
| `sacremoses` | Required by Helsinki opus-mt |
| `numpy` | Vector math for similarity |
| `pytest` | Tests (dev dependency) |

Install via:
```bash
uv add transformers torch sentence-transformers sentencepiece sacremoses numpy
uv add --dev pytest
```

**Note on torch:** `uv add torch` picks the platform default. On Windows/Mac/Linux CPU it should pull the CPU wheel automatically. If install fails, fall back to explicit CPU index URL (document this in README troubleshooting section).

---

## 6. Shared contract — `lab/contract.py`

Every use case script returns a dictionary matching this exact schema. The runner consumes only this. Define helper builders here.

### Result schema (one per use case)

```python
{
  "use_case_id": "01_classify_email_department",       # str, matches filename stem
  "type": "zero_shot_classification",                  # str, one of the five categories below
  "description": "Route incoming internal emails to the correct department.",
  "domain_relevance": "Reduces manual triage in a multi-department retail org.",
  "model": "facebook/bart-large-mnli",                 # HF model id
  "library": "transformers",                           # 'transformers' or 'sentence-transformers'
  "model_load_time_s": 4.21,                           # float, seconds
  "test_cases": [
    {
      "input": "...",                                  # str or dict (use case dependent)
      "expected": "IT",                                # any JSON-serialisable
      "actual": "IT",                                  # any JSON-serialisable
      "passed": True,                                  # bool
      "inference_time_s": 0.34,
      "notes": ""                                      # optional explanation, e.g. soft match details
    },
    ...
  ],
  "pass_count": 3,
  "total_count": 3,
  "pass_rate": 1.0,
  "total_inference_time_s": 1.02,
  "total_runtime_s": 5.23,                             # load + inference
  "error": null                                        # str if the script crashed, else null
}
```

### Type values (controlled vocabulary)
- `zero_shot_classification`
- `sentiment`
- `ner`
- `semantic_similarity`
- `language_detection`
- `translation`

### Helper in `contract.py`
Provide a `build_result(...)` function that takes the raw test results and computes the derived fields (`pass_count`, `pass_rate`, totals). This keeps each script lean.

---

## 7. Per-script contract

Every script in `usecases/` must:

1. Have a top-level docstring explaining the use case + travel-retail relevance.
2. Define `USE_CASE_ID`, `MODEL_ID`, `TASK_TYPE` as module constants.
3. Define `TEST_CASES` as a module-level list (so it's discoverable for testing).
4. Implement `run() -> dict` returning the schema in §6.
5. Have an `if __name__ == "__main__":` block that calls `run()` and prints the result as JSON (UTF-8, ensure_ascii=False).
6. Be runnable in isolation: `uv run python usecases/04_sentiment_customer_review.py` must work and print valid JSON to stdout.

**Template** (every script follows this):
```python
"""
Use case: <one-liner>
Travel-retail relevance: <one-liner>
Model: <model id>
Library: <transformers | sentence-transformers>
"""

import json
import time
from lab.contract import build_result

USE_CASE_ID = "04_sentiment_customer_review"
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
TASK_TYPE = "sentiment"

TEST_CASES = [
    {"input": "...", "expected": "positive"},
    {"input": "...", "expected": "neutral"},
    {"input": "...", "expected": "negative"},
]

def run() -> dict:
    # 1. Time the model load
    t0 = time.perf_counter()
    # ... load model / pipeline ...
    load_time = time.perf_counter() - t0

    # 2. Run each test case, time the inference
    results = []
    for tc in TEST_CASES:
        t1 = time.perf_counter()
        # ... predict ...
        actual = ...
        passed = (actual == tc["expected"])
        results.append({
            "input": tc["input"],
            "expected": tc["expected"],
            "actual": actual,
            "passed": passed,
            "inference_time_s": round(time.perf_counter() - t1, 4),
            "notes": "",
        })

    return build_result(
        use_case_id=USE_CASE_ID,
        type=TASK_TYPE,
        description="...",
        domain_relevance="...",
        model=MODEL_ID,
        library="transformers",
        model_load_time_s=round(load_time, 4),
        test_cases=results,
    )

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
```

---

## 8. The 11 use cases — specs and test data

For each, the test data below is **authoritative**. Use it verbatim. Do not "improve" or expand it during the first pass.

### Category A — Zero-shot Classification (3 scripts)

Model: **`facebook/bart-large-mnli`**
Library: `transformers`, pipeline `"zero-shot-classification"`
Evaluation: `actual == expected` (exact match on top-1 label).

---

#### 01 — `01_classify_email_department.py`
- **Use case:** Route internal emails to the correct department.
- **Relevance:** Multi-department retail operations get hundreds of emails; auto-routing cuts triage time.
- **Candidate labels:** `["IT", "Planner", "Pricing", "Warehouse", "Finance", "HR"]`
- **Test cases:**
  1. `"Server is down, customers can't checkout"` → `"IT"`
  2. `"Need to reorder fragrance stock for the airport store, running low"` → `"Planner"`
  3. `"Q3 expense report due Friday"` → `"Finance"`

---

#### 02 — `02_classify_feedback_topic.py`
- **Use case:** Categorise customer feedback by topic.
- **Relevance:** Travel-retail customer experience teams triage thousands of post-purchase comments.
- **Candidate labels:** `["product_quality", "staff_service", "pricing", "store_experience", "checkout_process"]`
- **Test cases:**
  1. `"The bottle of whisky I bought had a damaged seal"` → `"product_quality"`
  2. `"Cashier was rude when I asked for the tax-free form"` → `"staff_service"`
  3. `"Too expensive compared to my local market"` → `"pricing"`

---

#### 03 — `03_classify_supplier_urgency.py`
- **Use case:** Tag inbound supplier emails by intent / urgency.
- **Relevance:** Procurement teams handle dispute, delivery, invoice and routine emails in one inbox.
- **Candidate labels:** `["urgent", "routine", "dispute", "delivery_update", "invoice_query"]`
- **Test cases:**
  1. `"PO #12345 was overcharged by $500, please refund"` → `"dispute"`
  2. `"Please find attached the monthly catalogue update"` → `"routine"`
  3. `"Container delayed at customs, ETA pushed two weeks"` → `"delivery_update"`

---

### Category B — Sentiment Analysis (2 scripts)

Model: **`cardiffnlp/twitter-roberta-base-sentiment-latest`**
Library: `transformers`, pipeline `"sentiment-analysis"`
Model returns labels `LABEL_0` (negative), `LABEL_1` (neutral), `LABEL_2` (positive) OR `negative`/`neutral`/`positive` depending on config — normalise to lowercase string in the script.

---

#### 04 — `04_sentiment_customer_review.py`
**This is the FIRST script to build. Validate the whole pipeline on this one before writing any other use case.**
- **Use case:** Classify a customer review as positive / neutral / negative.
- **Relevance:** Volume of post-purchase reviews makes manual sentiment scoring impossible.
- **Test cases:**
  1. `"Amazing perfume selection, staff was so helpful!"` → `"positive"`
  2. `"It was okay, nothing special"` → `"neutral"`
  3. `"Waited 30 mins at checkout, never coming back"` → `"negative"`

---

#### 05 — `05_sentiment_internal_escalation.py`
- **Use case:** Flag internal emails with negative tone as escalation candidates.
- **Relevance:** Catch operational frustration before it becomes a manager escalation.
- **Mapping:** model output `negative` → `expected: "escalated"`; `positive` or `neutral` → `expected: "not_escalated"`
- **Test cases:**
  1. `"Quick question on SKU mapping when you have time"` → `"not_escalated"`
  2. `"Third time I'm raising this, no one is responding"` → `"escalated"`
  3. `"Update: warehouse received the shipment"` → `"not_escalated"`

---

### Category C — Named Entity Recognition (2 scripts)

Model: **`dslim/bert-base-NER`**
Library: `transformers`, pipeline `"ner"` with `aggregation_strategy="simple"`
Evaluation: extract `(entity_text, entity_type)` set. Compare against expected set with **set-overlap precision/recall**. Pass = F1 ≥ 0.66.

> **Honest limitation:** This model knows only PER / ORG / LOC / MISC. It will not tag product SKUs. Brand names like "Hermès" may be tagged ORG or MISC inconsistently. Test data uses only entities the model can realistically catch.

---

#### 06 — `06_ner_brands_locations.py`
- **Use case:** Extract brand and location mentions from customer comments.
- **Relevance:** Powers downstream analytics (which brands, which stores get most attention).
- **Test cases:**
  1. Input: `"Bought Chanel and Hermes at Changi yesterday"`
     Expected (set): `{("Chanel", "ORG"), ("Hermes", "ORG"), ("Changi", "LOC")}`
  2. Input: `"Macallan was sold out at the Hong Kong store"`
     Expected: `{("Macallan", "ORG"), ("Hong Kong", "LOC")}`
  3. Input: `"Picked up Dior and Estee Lauder at LAX"`
     Expected: `{("Dior", "ORG"), ("Estee Lauder", "ORG"), ("LAX", "LOC")}`

> Note: use ASCII forms ("Hermes", "Estee Lauder") in test data — the model's WordPiece tokenizer handles them more reliably than accented forms. If you discover the model output differs by a single token boundary (e.g. predicts "Estee" only), update `notes` with the actual output and keep the test honest.

---

#### 07 — `07_ner_people_orgs_procurement.py`
- **Use case:** Pull people and organisation names from procurement emails.
- **Relevance:** Auto-populate CRM and supplier contact records.
- **Test cases:**
  1. Input: `"John from Acme Logistics confirmed delivery"`
     Expected: `{("John", "PER"), ("Acme Logistics", "ORG")}`
  2. Input: `"Please contact Sarah Tan at Globex regarding the new collection"`
     Expected: `{("Sarah Tan", "PER"), ("Globex", "ORG")}`
  3. Input: `"Diageo representative will visit our Singapore office Monday"`
     Expected: `{("Diageo", "ORG"), ("Singapore", "LOC")}`

> All company names above are either generic or fictional. Do not substitute real luxury brands here.

---

### Category D — Semantic Similarity (2 scripts)

Model: **`sentence-transformers/all-MiniLM-L6-v2`**
Library: `sentence-transformers`
Evaluation: top-1 retrieved doc id matches expected. Pass = exact id match.

---

#### 08 — `08_similarity_complaint_lookup.py`
- **Use case:** Given a new complaint, retrieve the most similar past complaint.
- **Relevance:** Customer service can reuse previous resolutions.
- **Corpus** (define as `CORPUS` constant, list of `{"id": str, "text": str}`):
  ```
  c1: "Got a damaged cosmetic item"
  c2: "Cashier was rude at the HKG store"
  c3: "Double-charged on my credit card"
  c4: "Forgot to claim tax refund at the airport"
  c5: "Item different from website description"
  c6: "Long queue at checkout during peak hours"
  ```
- **Test cases:**
  1. Query: `"My perfume bottle arrived broken"` → expected: `"c1"`
  2. Query: `"Staff was impolite at HKG store"` → expected: `"c2"`
  3. Query: `"Charged twice for the same purchase"` → expected: `"c3"`

---

#### 09 — `09_similarity_sop_retrieval.py`
- **Use case:** Match a customer question to the most relevant internal SOP.
- **Relevance:** Front-line staff find the right policy quickly.
- **Corpus:**
  ```
  sop_returns:        "Return policy: items can be returned within 30 days with receipt"
  sop_refunds:        "Refund process for online and in-store purchases"
  sop_tax_free:       "How customers claim tax-free refunds at the airport"
  sop_lost_receipt:   "Procedure when a customer cannot produce a receipt"
  sop_damaged_goods:  "Steps to handle a customer reporting a damaged purchase"
  ```
- **Test cases:**
  1. Query: `"How do I claim a tax refund?"` → expected: `"sop_tax_free"`
  2. Query: `"What if I lose my receipt?"` → expected: `"sop_lost_receipt"`
  3. Query: `"Item arrived broken, what's the process?"` → expected: `"sop_damaged_goods"`

---

### Category E — Multilingual (2 scripts)

---

#### 10 — `10_language_detection.py`
Model: **`papluca/xlm-roberta-base-language-detection`**
Library: `transformers`, pipeline `"text-classification"`
Evaluation: top-1 ISO 639-1 code matches expected.
- **Use case:** Detect which language a customer message is in.
- **Relevance:** Airport customer base is multilingual; route to right-language support.
- **Test cases:**
  1. `"我在新加坡买了一瓶酒"` → `"zh"`
  2. `"ありがとうございました"` → `"ja"`
  3. `"I want to claim my tax refund"` → `"en"`

---

#### 11 — `11_translate_zh_en.py`
Model: **`Helsinki-NLP/opus-mt-zh-en`**
Library: `transformers`, pipeline `"translation"`
Evaluation: cosine similarity between model output and reference English, using `all-MiniLM-L6-v2` (load both models in this script). Pass threshold: **cosine ≥ 0.65**.

- **Use case:** Translate Chinese customer messages to English for triage.
- **Relevance:** Significant traveller segments write feedback in their native language.
- **Test cases:**
  1. Input: `"请问退税柜台在哪里？"` → reference: `"Where is the tax refund counter?"`
  2. Input: `"我想退货"` → reference: `"I want to return the goods"`
  3. Input: `"这瓶香水多少钱？"` → reference: `"How much is this bottle of perfume?"`

`notes` field for each test case must include the actual translation and the cosine score (e.g. `"actual='Where is the tax rebate counter?' cosine=0.81"`).

---

## 9. `run_all.py` — runner spec

### Behaviour
1. Discover all scripts matching `usecases/[0-9][0-9]_*.py`, sorted by filename.
2. For each script: run as a subprocess (`uv run python usecases/NN_*.py`), capture stdout, wall-clock total runtime.
3. Parse stdout as JSON. If parsing fails or exit code ≠ 0, record an error result with `error` populated and `passed=False` for all listed test cases (use `TEST_CASES` length if reachable, else 0).
4. Aggregate into `output/results.json` with a top-level wrapper:
   ```json
   {
     "run_metadata": {
       "started_at": "2026-06-11T14:30:00Z",
       "finished_at": "...",
       "total_wall_time_s": 312.4,
       "host_os": "Windows-11",
       "python_version": "3.11.x"
     },
     "use_cases": [ { schema from §6 }, ... ],
     "summary": {
       "total_use_cases": 11,
       "successful_use_cases": 10,
       "total_test_cases": 33,
       "total_passed": 27,
       "overall_pass_rate": 0.818
     }
   }
   ```
5. After writing `results.json`, call `lab.report.render(results)` to generate `output/report.html`.

### CLI
```
uv run python run_all.py                # run all
uv run python run_all.py --only 04      # run only use case 04
uv run python run_all.py --skip-html    # run all, skip report generation
```

### Execution mode
**Sequential subprocesses.** Reasons:
- Predictable memory (each model unloads when subprocess exits)
- Failure isolation
- One file = one use case is preserved
- Acceptable runtime (~5–8 min total on a modern CPU)

Do not parallelise. If overall time exceeds 15 min on the developer machine, document it in README rather than refactoring.

---

## 10. HTML report — `lab/report.py`

Single function: `render(results: dict, output_path: Path) -> None`

### Output: one self-contained `output/report.html` file.
- Inline CSS only (no external stylesheets, no CDN links)
- No JavaScript required for static viewing
- Optional: a tiny vanilla JS toggle to expand/collapse each use case section (no framework)

### Sections (in order)
1. **Header:** report title, run timestamp, host info, total wall time
2. **Summary cards row:** total use cases, successful use cases, total test cases, overall pass rate
3. **Headline table — model comparison:**
   | # | Use Case | Type | Model | Load (s) | Avg inference (s) | Pass rate |
4. **Per-use-case sections (one card per use case):**
   - Title bar with pass/fail badge (green ≥ 0.66, amber 0.33–0.65, red < 0.33)
   - Metadata: description, relevance, model, library, total runtime
   - Test case table: input (truncated to 200 chars with full text on hover/title attr), expected, actual, passed, inference time, notes
5. **Footer:** "Generated by `nlp-offline-lab` — pretrained models, no LLM API calls."

### Styling
- Neutral palette (white background, dark grey text)
- Monospace font for inputs/outputs
- Green `#16a34a` for pass, red `#dc2626` for fail, amber `#d97706` for borderline
- Readable on a laptop screen without horizontal scroll
- No emojis

### Visible disclosure (required)
Insert a fixed note near the top:
> Pass rates reflect zero-shot performance with no fine-tuning. Honest baseline on a tiny test set (3 cases per use case) — not a benchmark.

---

## 11. Tests — `tests/`

Use `pytest`. Minimal but real.

### `tests/test_runner.py`
- **Test:** runner discovers all 11 use case files in `usecases/`.
- **Test:** runner handles a subprocess that exits non-zero — records an error result, does not crash.
- **Test:** runner handles a subprocess that prints malformed JSON — records an error result with `error` field populated.
- **Test:** the produced `results.json` validates against the schema in §6 (key presence, types).

For the failure-injection tests, do NOT modify real use case files. Create a tiny fixture script in a temp dir and point a helper at it. Mock or parametrise `subprocess.run` if cleaner.

### `tests/test_report.py`
- **Test:** `render()` with a sample 2-use-case results dict produces an HTML file.
- **Test:** output HTML contains the use case ids, the summary numbers, and a `<table>` element.
- **Test:** output HTML is valid enough that `html.parser` can parse it without errors.

### Run via
```bash
uv run pytest -q
```

All tests must pass before the final commit.

---

## 12. Build sequence (mandatory order)

Do not skip steps. Each step must verify before proceeding to the next.

**Step 1 — Scaffold**
- Create folder, `git init`, `gh repo create`, `uv init`, `.gitignore`, empty `README.md`
- `uv add` all dependencies (§5)
- Commit + push: `chore: initial scaffold`

**Step 2 — Shared contract**
- Implement `lab/__init__.py` and `lab/contract.py`
- `build_result()` helper computes derived fields
- Commit + push: `feat: add shared utils and use case contract`

**Step 3 — Template script (use case 04)**
- Implement `04_sentiment_customer_review.py` end-to-end
- Run it standalone: `uv run python usecases/04_sentiment_customer_review.py`
- Verify it prints valid JSON matching §6 schema
- Verify at least one test case actually passes (sanity)
- If model output labels differ from `positive`/`neutral`/`negative`, normalise inside the script
- Commit + push: `feat: implement use case 04 sentiment customer review`

**Step 4 — Replicate to other scripts**
- Implement 01, 02, 03 (zero-shot)
- Implement 05 (escalation, reuses 04's model)
- Implement 06, 07 (NER)
- Implement 08, 09 (similarity)
- Implement 10, 11 (multilingual)
- After each script: run it standalone. If it crashes or all three test cases fail, see §13 (iterative correction).
- Commit + push at the milestones listed in §3.

**Step 5 — Runner**
- Implement `run_all.py`
- Test: `uv run python run_all.py --only 04` → produces results.json with one use case
- Test: `uv run python run_all.py` → produces full results.json
- Commit + push: `feat: add run_all runner`

**Step 6 — HTML report**
- Implement `lab/report.py`
- Test: open `output/report.html` in a browser (or validate via `html.parser`)
- Commit + push: `feat: add HTML report generator`

**Step 7 — Unit tests**
- Implement `tests/test_runner.py` and `tests/test_report.py`
- `uv run pytest -q` → all green
- Commit + push: `test: add runner and report unit tests`

**Step 8 — README**
- Write `README.md` (see §15)
- Commit + push: `docs: finalise README with usage and results summary`

**Step 9 — Reality check**
- Run `uv run python run_all.py` one more time end-to-end
- Open `output/report.html`, eyeball it
- Confirm summary numbers in HTML match `results.json`
- Confirm no real company names anywhere (`git grep -i <forbidden_terms>` if needed)
- If discrepancies found, fix and commit: `fix: ...`

---

## 13. Iterative correction protocol

When a use case script doesn't produce expected results, follow this loop. Do not modify test data to force passes. Do not silently lower thresholds.

For each failing test case:

1. **Reproduce in isolation.** Run the single script. Look at the raw model output, not just the pass/fail.
2. **Diagnose the root cause.** Choose one:
   - (a) **Bug in script logic** — wrong label normalisation, wrong field used from pipeline output, off-by-one. → Fix script.
   - (b) **Model genuinely disagrees** with expected, and the disagreement is reasonable — e.g. it labels "neutral" where you expected "negative" but the input is mild. → Either:
     - Accept the failure and add a `notes` entry explaining the model's behaviour, OR
     - Adjust the expected value if the model's answer is the more defensible interpretation (document why in commit message)
   - (c) **Model is wrong and the input is fair** — note it as a model limitation; leave the test failing; the report will surface it honestly.
3. **Cap iterations at 3 per test case.** If still failing after 3 attempts at (a), default to (c).
4. **Never change a test input to make a bad model look good.**

For NER (use cases 06, 07): if predicted entities differ by a token boundary (e.g. "Estee" vs "Estee Lauder"), record both expected and actual sets verbatim in the result and let F1 reflect reality.

For translation (use case 11): if cosine < 0.65, do not lower the threshold. Note the actual cosine and the model's output in `notes`. Failure is acceptable; honesty is mandatory.

---

## 14. Acceptance criteria (when is "done"?)

All of these must hold simultaneously:

1. ✅ All 11 scripts execute without raising uncaught exceptions
2. ✅ `uv run python run_all.py` produces valid `output/results.json` matching §6 schema
3. ✅ `output/report.html` renders correctly in a browser, summary numbers consistent with `results.json`
4. ✅ All unit tests pass (`uv run pytest -q`)
5. ✅ Git history reflects the milestone commits from §3, all pushed to `origin/main`
6. ✅ At least one use case achieves `pass_rate == 1.0` on its 3 test cases (proof the stack actually works end-to-end)
7. ✅ Total wall time of `run_all.py` is documented in README
8. ✅ Zero mentions of real-world specific company names anywhere in the repo

Pass rates per use case are NOT acceptance gates — they're outcomes to surface honestly in the report.

---

## 15. `README.md` — final contents

Sections:
1. **What this is** — one paragraph on offline pretrained NLP for travel-retail PoC
2. **Why offline models** — short rationale (privacy, budget, latency)
3. **The 11 use cases** — table: id, name, type, model
4. **Quick start**
   ```bash
   git clone <repo>
   cd nlp-offline-lab
   uv sync
   uv run python run_all.py
   # open output/report.html
   ```
5. **First run note** — models auto-download to `~/.cache/huggingface` (~4–5GB)
6. **Running a single use case** — `uv run python usecases/04_sentiment_customer_review.py`
7. **Results summary** — paste the latest `pass_rate` numbers as a table
8. **Honest limitations** — bullets pulled from §13 + the model-specific caveats in §8
9. **Troubleshooting**
   - `torch` install issue on Windows → try CPU index URL
   - `sentencepiece` missing → `uv add sentencepiece`
   - Out of disk → list which models can be skipped
10. **Tests** — `uv run pytest -q`
11. **License** — MIT (use a stock MIT LICENSE file)

---

## 16. Known limitations to flag honestly in the report and README

- All evaluations on 3 test cases each — illustrative, not statistically meaningful
- Zero-shot classifiers struggle when candidate labels overlap conceptually
- BERT-NER does not recognise products / SKUs / accented brand spellings reliably
- Twitter-RoBERTa sentiment was trained on Twitter; degrades on long formal text
- Helsinki opus-mt translates word-for-word in some cases; nuance is lost
- Language detection top-1 may be wrong on very short inputs (< 3 words)
- None of these models are fine-tuned on travel-retail jargon — that would be the next step

---

## 17. What is explicitly out of scope

- Fine-tuning any model
- GPU / CUDA paths
- Streaming / batch inference
- Async runner
- Web UI (HTML report is static, read-only)
- Real customer data (use only the test data in §8)
- Production logging / observability
- Dockerisation
- CI/CD pipeline

---

## 18. Final checklist for the executing agent

Before declaring the build complete:

- [ ] Repo created, public, pushed to GitHub
- [ ] `uv sync` works from a fresh clone
- [ ] All 11 scripts runnable standalone
- [ ] `run_all.py` produces `results.json` and `report.html`
- [ ] `pytest` passes
- [ ] README finalised with actual pass rates from the latest run
- [ ] No real company / brand identifying anything proprietary anywhere
- [ ] Commit history matches the milestone plan
- [ ] Disclosure note about pass-rate honesty present in HTML report
- [ ] Reality check pass done (§12 step 9)

Once all boxes are ticked, post a short summary to the user: total runtime, per-use-case pass rates, link to the report file, and any honest failures worth their attention.

---

*End of plan.*
