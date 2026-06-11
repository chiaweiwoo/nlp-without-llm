# Offline NLP Learning Curriculum

## Summary

The repository is being redesigned around:

> **formal NLP problem -> technical method -> luxury-retail application**

Each study item is one runnable problem-method combination with exactly **3
evaluation examples**. Supporting corpora, catalogues, policies, OCR assets, and
few-shot support sets are committed separately and do not count as evaluation
examples.

## Stages

### 1. Documentation first

- Update `README.md` and `AGENTS.md`.
- Record the target curriculum, artifact policy, and execution model here.

### 2. Implementation

- Replace the old scenario-first studies with the new curriculum.
- Prepare committed fixtures under `data/`.
- Run studies successfully and save validated outputs.

### 3. Aggregation

- Combine individual results into one aggregate JSON file.
- Render one final HTML report grouped by formal problem.

## Principles

- One study = one **problem + technique + application** combination.
- Multiclass classification uses at least **3 classes**.
- Compared techniques must share the same evaluation examples.
- Scripts must be runnable; no fabricated successful outputs.
- A successful stored result is a valid stopping point until its code, fixture,
  expected output, model, or configuration changes.

## Target Curriculum

| IDs | Formal problem | Technical methods | Luxury-retail application |
|---|---|---|---|
| `01` | Language Identification | Transformer classification | Identify traveller enquiry language |
| `02` | Machine Translation | Encoder-decoder transformer | Translate Mandarin luxury-retail enquiries |
| `03-04` | Sentiment Analysis | Lexicon scoring, transformer classifier | Evaluate boutique review sentiment |
| `05-06` | Named Entity Recognition | Gazetteer/rules, transformer token classification | Extract brands, people, organisations, airports, locations |
| `07-09` | Multiclass Text Classification | Keyword rules, zero-shot NLI, few-shot embedding prototypes | Route incidents into inventory, client service, loss prevention |
| `10-12` | Information Retrieval | BM25, dense bi-encoder, cross-encoder reranking | Retrieve SOPs for returns, authenticity, missed collections |
| `13` | Extractive Question Answering | Transformer span extraction | Extract exact answers from policy text |
| `14` | Template-Based Information Extraction | Regex and deterministic parsing | Parse receipts and stock messages |
| `15-16` | Entity Resolution | Fuzzy matching, dense catalogue matching | Resolve product aliases and misspellings to SKUs |
| `17` | Natural Language Inference | Three-way NLI classification | Detect promotion-report entailment, neutrality, contradiction |
| `18` | Extractive Summarization | Sentence ranking | Summarize shift handovers |
| `19` | Optical Character Recognition | Transformer OCR | Read synthetic luxury-retail receipts |

## Shared Evaluation Design

### Classification

Shared classes:

- `inventory_exception`
- `client_service_exception`
- `loss_prevention_exception`

Shared evaluation examples:

1. The limited-edition handbag shows available online, but the boutique has no stock.
2. A VIP traveller has waited forty minutes for a preorder collection.
3. A diamond bracelet is missing after the evening stock count.

### Retrieval

Shared SOP retrieval questions:

1. Can a traveller return an unopened perfume without the original receipt?
2. What should staff do when a luxury watch authenticity card is missing?
3. Can an airport preorder be collected after the passenger's flight departs?

### Entity Resolution

Shared queries:

1. Exact alias
2. Abbreviation
3. Misspelling

## Storage Policy

Commit:

- synthetic fixtures under `data/`
- compact per-study JSON result files
- aggregate `output/results.json`
- aggregate `output/report.html`
- OCR fixture images and metadata

Do not commit:

- model weights
- caches
- virtual environments
- temporary intermediates

Artifacts should stay small enough for normal GitHub use. Favor compact JSON,
small images, and minimal synthetic corpora.

## Execution Model

Target commands:

```bash
uv run python run_study.py --list
uv run python run_study.py --only 07
uv run python run_study.py --problem information_retrieval
uv run python run_study.py --all --max-tier medium
uv run python run_study.py --report-only
```

Runtime tiers:

- `fast`: rules, fuzzy matching, TF-IDF, BM25
- `medium`: embeddings, NER, QA, reranking, summarization
- `heavy`: translation and OCR

All studies remain fully implemented and runnable.

## Result Contract Direction

Future result records should include:

- `problem_name`
- `technique_name`
- `application_name`
- `comparison_group`
- `runtime_tier`
- `primary_metric`
- `metrics`
- `completed_at`

## Migration Notes

The current `usecases/` directory is still the previous study generation. It
contains useful source material but does not match the target curriculum yet.

The first implementation tasks after this document are:

1. rework the result contract
2. rework the runner interface
3. add `data/` fixtures
4. replace redundant scripts
5. regenerate compact committed outputs
