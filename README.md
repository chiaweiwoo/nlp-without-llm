# nlp-without-llm

A self-contained portfolio and learning repository demonstrating the use of **offline pretrained NLP models** (no external LLM APIs, no internet connection required at inference time) on practical **travel-retail and duty-free** business scenarios. Each use case is implemented as an isolated, standalone script, allowing you to explore the exact code, library interface, and model behaviors in complete isolation.

---

## Why Offline Models?

While modern LLM APIs (OpenAI, Anthropic, Gemini) are highly capable, they introduce significant challenges for certain enterprise environments:
- **Data Privacy & Compliance:** Customer feedback, internal procurement emails, and operational SOPs often contain sensitive information that cannot leave local systems.
- **Cost Efficiency:** Processing thousands of customer reviews or emails daily via API tokens can lead to high recurring costs.
- **Latency & Reliability:** Airport retail operations require low-latency predictions and must remain fully functional even when local network/internet connectivity is degraded.

By deploying small, task-specific models locally on CPU, you gain full ownership over inference latency, data security, and operational uptime with zero ongoing API licensing or usage costs.

---

## The 11 Use Cases

The lab covers 6 different NLP tasks across 11 specific duty-free retail scenarios:

| ID | Use Case Name | Task Type | Pretrained Model |
|---|---|---|---|
| **01** | [Classify Email Department](usecases/01_classify_email_department.py) | `zero_shot_classification` | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| **02** | [Classify Feedback Topic](usecases/02_classify_feedback_topic.py) | `zero_shot_classification` | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| **03** | [Classify Supplier Urgency](usecases/03_classify_supplier_urgency.py) | `zero_shot_classification` | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| **04** | [Sentiment Customer Review](usecases/04_sentiment_customer_review.py) | `sentiment` | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **05** | [Sentiment Internal Escalation](usecases/05_sentiment_internal_escalation.py) | `sentiment` | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **06** | [NER Brands & Locations](usecases/06_ner_brands_locations.py) | `ner` | `dslim/bert-base-NER` |
| **07** | [NER People & Orgs Procurement](usecases/07_ner_people_orgs_procurement.py) | `ner` | `dslim/bert-base-NER` |
| **08** | [Similarity Complaint Lookup](usecases/08_similarity_complaint_lookup.py) | `semantic_similarity` | `sentence-transformers/all-mpnet-base-v2` |
| **09** | [Similarity SOP Retrieval](usecases/09_similarity_sop_retrieval.py) | `semantic_similarity` | `sentence-transformers/all-mpnet-base-v2` |
| **10** | [Language Detection](usecases/10_language_detection.py) | `language_detection` | `papluca/xlm-roberta-base-language-detection` |
| **11** | [Translate ZH to EN](usecases/11_translate_zh_en.py) | `translation` | `facebook/nllb-200-distilled-600M` |

---

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (fast Python package installer and resolver)

### Installation & Execution
1. Clone the repository:
   ```bash
   git clone https://github.com/chiaweiwoo/nlp-without-llm.git
   cd nlp-without-llm
   ```
2. Synchronize virtual environment and dependencies:
   ```bash
   uv sync
   ```
3. Run the evaluation runner:
   ```bash
   uv run python run_all.py
   ```
4. View results:
   Open `output/report.html` in your favorite web browser to inspect the visual, interactive results dashboard.

> **First Run Note:** Spawning the runner for the first time will automatically download the required model weights (totaling ~4–5 GB) to your local Hugging Face cache folder (`~/.cache/huggingface`). Subsequent runs will run entirely offline and start instantly.

---

## Running a Single Use Case

Every use case script is self-contained and can be executed in complete isolation:

```bash
uv run python usecases/04_sentiment_customer_review.py
```

It will execute the test cases and output a standardized JSON result to stdout.

---

## Results Summary

Below are the outcomes from the latest evaluation run executed on a standard CPU:

- **Total Execution Time (Wall-clock):** ~275 seconds (approx. 4.6 minutes)
- **Overall Pass Rate:** **90.91%** (30/33 test cases passed)

### Per-Use-Case Evaluation

| ID | Use Case | Pass Rate | Inferences Passed | Primary Model |
|---|---|---|---|---|
| **01** | Classify Email Department | **66.67%** | 2 / 3 | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| **02** | Classify Feedback Topic | **66.67%** | 2 / 3 | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| **03** | Classify Supplier Urgency | **100.0%** | 3 / 3 | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| **04** | Sentiment Customer Review | **100.0%** | 3 / 3 | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **05** | Sentiment Internal Escalation | **100.0%** | 3 / 3 | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **06** | NER Brands & Locations | **66.67%** | 2 / 3 | `dslim/bert-base-NER` |
| **07** | NER People & Orgs Procurement | **100.0%** | 3 / 3 | `dslim/bert-base-NER` |
| **08** | Similarity Complaint Lookup | **100.0%** | 3 / 3 | `sentence-transformers/all-mpnet-base-v2` |
| **09** | Similarity SOP Retrieval | **100.0%** | 3 / 3 | `sentence-transformers/all-mpnet-base-v2` |
| **10** | Language Detection | **100.0%** | 3 / 3 | `papluca/xlm-roberta-base-language-detection` |
| **11** | Translate ZH to EN | **100.0%** | 3 / 3 | `facebook/nllb-200-distilled-600M` |

---

## Honest Limitations

Since these are small, general-purpose models operating zero-shot on specialized retail business domain data, you will notice certain limitations:
1. **Zero-Shot Classifiers (01, 02, 03):** Even with DeBERTa-v3, models can struggle on labels with fuzzy semantic boundaries (e.g., routing an "expense report" to `Planner` instead of `Finance` because of "Q3... due Friday").
2. **Tokenization & Subwords in NER (06):** Pretrained BERT NER expects standard word forms. Multi-word brand names like "Estee Lauder" can get split into subwords and misclassified as `PER` (person) instead of `ORG` (organization).
3. **Sentiment Nuance (04, 05):** Models trained on Twitter data handle standard feedback well, but might miss context-specific sarcasm or formal business jargon.
4. **Translation Cosine Scores (11):** While NLLB-200 provides highly accurate translations (e.g. translating `"我想退货"` to `"I want to return."`), minor variation from the exact reference translation ("I want to return the goods") is reflected in the cosine similarity scores, although all now pass our 0.65 threshold.
5. **Small Test Set:** Results are measured on a tiny benchmark (3 test cases per script) for illustrative purposes and do not constitute a statistically significant performance benchmark.

---

## Troubleshooting

### PyTorch installation issues on Windows CPU
If installing `torch` fails on Windows, you may need to specify the official PyTorch CPU wheel index url directly in your environment:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Missing sentencepiece / sacremoses tokenizer requirements
Some models require additional tokenizer packages. They are included in `pyproject.toml` dependencies, but if you run into import errors, manually add them to your environment:
```bash
uv add sentencepiece sacremoses
```

### Out of disk space
If downloading all model weights (~5GB) exceeds your disk space, you can skip heavy models (like the 2.4GB `facebook/nllb-200-distilled-600M` used in 11) by running the runner on only a specific use case:
```bash
uv run python run_all.py --only 04
```

---

## Tests

The project includes standard unit tests to verify the runner discovery, subprocess execution logic, and HTML rendering components.

Run the test suite using `pytest`:
```bash
uv run python -m pytest -q
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
