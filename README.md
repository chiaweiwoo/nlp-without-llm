# nlp-without-llm

An offline NLP study repository for **luxury retail** use cases. The repository
is being redesigned around a clearer structure:

> **formal NLP problem -> technical method -> luxury-retail application**

Each use case is a standalone script that can run locally on CPU after model
weights are cached. The goal is to learn tradeoffs between rules, classic NLP,
and pretrained transformer models without relying on hosted LLM APIs.

---

## Current Direction

The repository is moving from a scenario-first collection of scripts to a
problem-first study curriculum.

The redesign is tracked in [docs/study-plan.md](/C:/Users/chiaw/OneDrive/Desktop/playground/nlp-without-llm/docs/study-plan.md).

Implementation will proceed in three stages:

1. Update repository documentation and agent guidance.
2. Replace the old use-case set with the new problem-technique curriculum,
   prepare fixtures, and generate validated results.
3. Aggregate all individual results into one HTML report.

This means the current scripts are still the previous generation until the new
curriculum is fully implemented.

During implementation, the expectation is to continue through the requested
scope until it is fully implemented, tested, and summarized. Do not stop at
intermediate checkpoints unless the user explicitly says to pause.

---

## Study Principles

- Every study item is one **problem + technique + application** combination.
- Every study item has exactly **3 evaluation examples**.
- Multiclass classification should be at least **3 classes**.
- When two techniques solve the same problem, they must share the same
  evaluation inputs.
- Scripts must be **real and runnable**. No placeholder or fabricated success
  outputs are allowed.
- Once a study has been run successfully and its result is stored, that result
  is treated as a valid stopping point until the code, fixture, model, or
  expected output changes.

---

## Target Curriculum

The target redesign covers **12 formal problems** and **19 runnable
problem-method combinations**:

| IDs | Formal problem | Technical methods |
|---|---|---|
| `01` | Language Identification | Transformer classification |
| `02` | Machine Translation | Encoder-decoder transformer |
| `03-04` | Sentiment Analysis | Lexicon scoring, transformer classifier |
| `05-06` | Named Entity Recognition | Gazetteer/rules, transformer token classification |
| `07-09` | Multiclass Text Classification | Keyword rules, zero-shot NLI, few-shot prototype classification |
| `10-12` | Information Retrieval | BM25, dense bi-encoder, cross-encoder reranking |
| `13` | Extractive Question Answering | Transformer span extraction |
| `14` | Template-Based Information Extraction | Regex and deterministic parsing |
| `15-16` | Entity Resolution | Fuzzy matching, dense catalogue matching |
| `17` | Natural Language Inference | Three-way NLI classification |
| `18` | Extractive Summarization | Sentence ranking |
| `19` | Optical Character Recognition | Transformer OCR |

All examples, corpora, catalogues, policies, support sets, and OCR fixtures
will stay in the **luxury-retail domain**.

---

## Data And Artifacts

The repository will commit:

- synthetic input fixtures under `data/`
- OCR fixture images and metadata
- compact per-study result JSON files
- aggregate `output/results.json`
- aggregate `output/report.html`

The repository will not commit:

- model weights
- Hugging Face caches
- virtual environments
- temporary intermediates

Artifacts must stay compact enough for normal GitHub usage. The working target
is small synthetic datasets, small generated images, and concise JSON/HTML.

---

## Execution Model

The redesigned runner will support:

```bash
uv sync
uv run python -m pytest -q
uv run python run_study.py --list
uv run python run_study.py --only 07
uv run python run_study.py --problem information_retrieval
uv run python run_study.py --all --max-tier medium
uv run python run_study.py --report-only
```

Runtime tiers are scheduling controls, not fake implementations:

- `fast`: rules, fuzzy matching, TF-IDF, BM25, and similar lightweight methods
- `medium`: embeddings, NER, QA, reranking, summarization
- `heavy`: larger translation or OCR models

No-argument execution should become a **status-oriented** command rather than an
implicit full run.

---

## Current Repository State

The current `usecases/` directory still contains the previous study generation.
Those files remain useful as source material, but they do not yet match the
target curriculum described above.

The next implementation work will:

- update the result contract
- add shared fixture directories
- rework the runner and report grouping
- replace redundant scenario scripts with the new curriculum

---

## Tests

Use `uv` for dependency and command execution:

```bash
uv sync
uv run python -m pytest -q
```

Unit tests should stay fast and must mock model boundaries where appropriate.

---

## License

MIT. See [LICENSE](LICENSE).
