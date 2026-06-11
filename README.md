# nlp-without-llm

A self-contained study repository showing how offline pretrained NLP models behave on travel-retail scenarios without hosted LLM APIs. Each use case is a standalone Python script, and the repo now centers on committed study artifacts rather than an all-or-nothing batch runner.

---

## Why Offline Models?

Offline inference is useful when:

- sensitive customer or operational text should stay local
- recurring API cost matters
- airport retail workflows need predictable behavior even with weak connectivity

This repo is not trying to automate a production pipeline. It is a study of what
small local models can and cannot do on CPU.

---

## Use Case Order

The use cases are ordered as a study flow rather than by implementation history:

| ID | Use Case Name | Task Type | Pretrained Model |
|---|---|---|---|
| **01** | [Language Detection](usecases/01_language_detection.py) | `language_detection` | `papluca/xlm-roberta-base-language-detection` |
| **02** | [Translate ZH to EN](usecases/02_translate_zh_en.py) | `translation` | `facebook/nllb-200-distilled-600M` |
| **03** | [Sentiment Customer Review](usecases/03_sentiment_customer_review.py) | `sentiment` | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **04** | [Sentiment Internal Escalation](usecases/04_sentiment_internal_escalation.py) | `sentiment` | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **05** | [NER Brands & Locations](usecases/05_ner_brands_locations.py) | `ner` | `dslim/bert-base-NER` |
| **06** | [NER People & Orgs Procurement](usecases/06_ner_people_orgs_procurement.py) | `ner` | `dslim/bert-base-NER` |
| **07** | [Classify Email Department](usecases/07_classify_email_department.py) | `zero_shot_classification` | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| **08** | [Classify Supplier Urgency](usecases/08_classify_supplier_urgency.py) | `zero_shot_classification` | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| **09** | [Classify Feedback Topic](usecases/09_classify_feedback_topic.py) | `zero_shot_classification` | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| **10** | [Similarity Complaint Lookup](usecases/10_similarity_complaint_lookup.py) | `semantic_similarity` | `sentence-transformers/all-mpnet-base-v2` |
| **11** | [Similarity SOP Retrieval](usecases/11_similarity_sop_retrieval.py) | `semantic_similarity` | `sentence-transformers/all-mpnet-base-v2` |
| **12** | [Few-shot Email Department](usecases/12_fewshot_email_department.py) | `zero_shot_classification` | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| **13** | [Few-shot Supplier Urgency](usecases/13_fewshot_supplier_urgency.py) | `zero_shot_classification` | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| **14** | [Few-shot Feedback Topic](usecases/14_fewshot_feedback_topic.py) | `zero_shot_classification` | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

### Install

```bash
uv sync
```

### Run the study workflow

```bash
uv run python run_study.py
```

This does the following:

- reuses existing per-use-case result JSON files when they are valid
- runs only missing baseline scenarios
- records the few-shot scenarios `12` to `14` as known local limits instead of executing them
- rebuilds `output/results.json` and `output/report.html`

### Run a single use case directly

```bash
uv run python usecases/03_sentiment_customer_review.py
```

Successful stdout is one JSON document.

---

## Study Outputs

The study workflow writes local artifacts to `output/`:

- `output/result_<use_case>.json` for each scenario
- `output/results.json` for the aggregate
- `output/report.html` for the final report

These files are generated locally and should not be committed.

---

## Current Results

Latest local study summary:

- **Total use cases:** `14`
- **Successful local runs:** `11`
- **Total test cases:** `54`
- **Total passed:** `37`
- **Overall pass rate:** `68.52%`

### Per-use-case results

| ID | Use Case | Status | Pass Rate | Passed |
|---|---|---|---|---|
| **01** | Language Detection | `ok` | `100.0%` | `3 / 3` |
| **02** | Translate ZH to EN | `ok` | `100.0%` | `3 / 3` |
| **03** | Sentiment Customer Review | `ok` | `100.0%` | `3 / 3` |
| **04** | Sentiment Internal Escalation | `ok` | `100.0%` | `3 / 3` |
| **05** | NER Brands & Locations | `ok` | `66.67%` | `2 / 3` |
| **06** | NER People & Orgs Procurement | `ok` | `100.0%` | `3 / 3` |
| **07** | Classify Email Department | `ok` | `100.0%` | `5 / 5` |
| **08** | Classify Supplier Urgency | `ok` | `80.0%` | `4 / 5` |
| **09** | Classify Feedback Topic | `ok` | `100.0%` | `5 / 5` |
| **10** | Similarity Complaint Lookup | `ok` | `100.0%` | `3 / 3` |
| **11** | Similarity SOP Retrieval | `ok` | `100.0%` | `3 / 3` |
| **12** | Few-shot Email Department | `known_limit` | `0.0%` | `0 / 5` |
| **13** | Few-shot Supplier Urgency | `known_limit` | `0.0%` | `0 / 5` |
| **14** | Few-shot Feedback Topic | `known_limit` | `0.0%` | `0 / 5` |

### What the negative results mean

The three few-shot scenarios were not rerun. Prior CPU runs already showed they take more than 20 minutes because long few-shot NLI prompts make this setup impractical locally. In this study they are recorded as negative outcomes:

- local CPU inference is not practical for those scenarios
- a hosted LLM API is likely the better tool there

That conclusion is part of the point of the repo.

---

## Practical Observations

1. The baseline local models are strong on:
   - language detection
   - translation
   - standard sentiment
   - semantic similarity retrieval

2. NER is usable but imperfect.
   - `05_ner_brands_locations` still misses some entity structure and only passes at the threshold level.

3. Zero-shot NLI classification is viable locally, but expensive on CPU.
   - `07` and `08` each take around five minutes because every test case is scored against ten hypotheses.

4. Few-shot prompt expansion is where the local approach stops making sense.
   - `12` to `14` are intentionally preserved as negative study findings.

---

## Troubleshooting

### PyTorch installation issues on Windows CPU

If `torch` installation fails on Windows:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Tokenizer dependencies

If a tokenizer dependency is missing:

```bash
uv add sentencepiece sacremoses
```

### Hugging Face warnings

Some runs emit unauthenticated Hugging Face warnings during model resolution. They do not invalidate the saved results.

---

## Tests

Run the fast unit suite with:

```bash
uv run python -m pytest -q
```

Current suite status: `6 passed`

---

## License

MIT. See [LICENSE](LICENSE).
