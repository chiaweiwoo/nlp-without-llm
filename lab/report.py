import html
import json
from pathlib import Path

CODE_DETAILS = {
    "01_language_detection": {
        "loader": "pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')",
        "params": "labels inferred directly from model output",
        "call": "classifier(text)[0]",
    },
    "02_translate_zh_en": {
        "loader": "AutoTokenizer.from_pretrained(model, src_lang='zho_Hans')\nAutoModelForSeq2SeqLM.from_pretrained(model)\nSentenceTransformer('all-mpnet-base-v2')",
        "params": "forced_bos_token_id='eng_Latn', cosine threshold >= 0.65",
        "call": "model.generate(...)\nutil.cos_sim(actual_embedding, expected_embedding)",
    },
    "03_sentiment_customer_review": {
        "loader": "custom lexicon baseline",
        "params": "positive_terms, negative_terms",
        "call": "lexicon_sentiment(text, positive_terms, negative_terms)",
    },
    "04_sentiment_internal_escalation": {
        "loader": "pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')",
        "params": "normalize label to positive / neutral / negative",
        "call": "classifier(text)[0]",
    },
    "05_ner_brands_locations": {
        "loader": "custom gazetteer phrase matcher",
        "params": "ENTITY_PHRASES, F1 threshold >= 0.66",
        "call": "extract_entities(text)",
    },
    "06_ner_people_orgs_procurement": {
        "loader": "pipeline('ner', model='dslim/bert-base-NER', aggregation_strategy='simple')",
        "params": "entity-set F1 threshold >= 0.66",
        "call": "ner(text)",
    },
    "07_classify_email_department": {
        "loader": "custom keyword rules",
        "params": "CLASS_KEYWORDS for 3 classes",
        "call": "classify_with_rules(text)",
    },
    "08_classify_supplier_urgency": {
        "loader": "AutoTokenizer.from_pretrained(model)\nAutoModelForSequenceClassification.from_pretrained(model)",
        "params": "3 class-specific NLI hypotheses",
        "call": "softmax(model(tokenizer(text, hypothesis)))",
    },
    "09_classify_feedback_topic": {
        "loader": "SentenceTransformer('all-mpnet-base-v2', device='cpu')",
        "params": "3 support examples per class, centroid prototypes",
        "call": "util.cos_sim(query_embedding, class_prototype)",
    },
    "10_similarity_complaint_lookup": {
        "loader": "custom BM25 lexical retriever",
        "params": "k1=1.5, b=0.75",
        "call": "bm25_scores(query, index)",
    },
    "11_similarity_sop_retrieval": {
        "loader": "SentenceTransformer('all-mpnet-base-v2', device='cpu')",
        "params": "corpus pre-encoded once",
        "call": "util.cos_sim(query_embedding, corpus_embeddings)",
    },
    "12_fewshot_email_department": {
        "loader": "CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')",
        "params": "rerank top 3 BM25 candidates",
        "call": "reranker.predict([(query, candidate_text), ...])",
    },
    "13_fewshot_supplier_urgency": {
        "loader": "AutoTokenizer.from_pretrained(model)\nAutoModelForQuestionAnswering.from_pretrained(model)",
        "params": "manual argmax span extraction over start/end logits",
        "call": "tokenizer(question, context) -> model(**inputs)",
    },
    "14_fewshot_feedback_topic": {
        "loader": "custom regex parser",
        "params": "pattern per template family",
        "call": "extract_fields(text)",
    },
    "15_entity_resolution_fuzzy": {
        "loader": "custom fuzzy matcher",
        "params": "SequenceMatcher ratio over catalog names",
        "call": "fuzzy_ratio(query, catalog_name)",
    },
    "16_entity_resolution_dense": {
        "loader": "SentenceTransformer('all-mpnet-base-v2', device='cpu')",
        "params": "catalog encoded once",
        "call": "util.cos_sim(query_embedding, catalog_embeddings)",
    },
    "17_natural_language_inference": {
        "loader": "AutoTokenizer.from_pretrained(model)\nAutoModelForSequenceClassification.from_pretrained(model)",
        "params": "3-way NLI logits mapped to label names",
        "call": "softmax(model(tokenizer(premise, hypothesis)))",
    },
    "18_extractive_summarization": {
        "loader": "custom sentence ranker",
        "params": "priority_terms, top_k=2",
        "call": "rank_sentences_by_keywords(text, priority_terms)",
    },
    "19_optical_character_recognition": {
        "loader": "TrOCRProcessor.from_pretrained(model)\nVisionEncoderDecoderModel.from_pretrained(model)",
        "params": "generated receipt fixtures, image -> pixel_values",
        "call": "model.generate(pixel_values) -> processor.batch_decode(...)",
    },
}


def _pass_rate_class(pass_rate: float) -> str:
    if pass_rate >= 0.66:
        return "pass-text"
    if pass_rate >= 0.33:
        return "amber-text"
    return "fail-text"


def _status_badge(use_case: dict) -> tuple[str, str, str]:
    status = use_case.get("status", "failed")
    pass_count = use_case.get("pass_count", 0)
    total_count = use_case.get("total_count", 0)
    pass_rate = use_case.get("pass_rate", 0.0)

    if status == "known_limit":
        return "badge-fail", "KNOWN LIMIT", "badge-fail"
    if status == "timeout":
        return "badge-fail", "TIMEOUT", "badge-fail"
    if status == "failed":
        return "badge-fail", "ERROR", "badge-fail"
    if pass_rate >= 0.66:
        return "badge-pass", f"PASS ({pass_count}/{total_count})", "badge-pass"
    if pass_rate >= 0.33:
        return "badge-amber", f"BORDERLINE ({pass_count}/{total_count})", "badge-amber"
    return "badge-fail", f"FAIL ({pass_count}/{total_count})", "badge-fail"


def render(results: dict, output_path: Path = Path("output/report.html")) -> None:
    """Render a self-contained HTML report from aggregated study results."""
    run_metadata = results.get("run_metadata", {})
    summary = results.get("summary", {})
    use_cases = results.get("use_cases", [])

    overall_pass_rate_pct = f"{round(summary.get('overall_pass_rate', 0.0) * 100, 2)}%"

    comparison_rows = ""
    nav_links_html = ""
    use_case_details_html = ""

    for idx, uc in enumerate(use_cases, 1):
        uc_id = uc.get("use_case_id", "")
        code_details = CODE_DETAILS.get(
            uc_id,
            {
                "loader": "N/A",
                "params": "N/A",
                "call": "N/A",
            },
        )
        problem_name = uc.get("problem_name", uc.get("type", "unknown"))
        technique_name = uc.get("technique_name", uc.get("library", "unknown"))
        application_name = uc.get("application_name", uc.get("description", ""))
        runtime_tier = uc.get("runtime_tier", "medium")
        primary_metric = uc.get("primary_metric", "pass_rate")
        metrics = uc.get("metrics", {})
        model = uc.get("model", "")
        load_time = uc.get("model_load_time_s", 0.0)
        pass_rate = uc.get("pass_rate", 0.0)
        pass_rate_str = f"{round(pass_rate * 100, 1)}%"
        completed_at = uc.get("completed_at") or "N/A"
        badge_class, badge_text, badge_dot_class = _status_badge(uc)

        test_cases = uc.get("test_cases", [])
        avg_inference = (
            sum(tc.get("inference_time_s", 0.0) for tc in test_cases) / len(test_cases)
            if test_cases
            else 0.0
        )

        comparison_rows += f"""
        <tr>
            <td class="col-small">{idx}</td>
            <td class="col-use-case"><strong>{html.escape(uc_id)}</strong></td>
            <td class="col-problem">{html.escape(problem_name)}</td>
            <td class="col-technique">{html.escape(technique_name)}</td>
            <td class="col-tier">{html.escape(runtime_tier)}</td>
            <td class="mono col-model" title="{html.escape(model)}">{html.escape(model)}</td>
            <td class="col-metric">{load_time:.3f}s</td>
            <td class="col-metric">{avg_inference:.4f}s</td>
            <td class="col-metric {_pass_rate_class(pass_rate)} font-bold">{pass_rate_str}</td>
        </tr>
        """

        nav_links_html += f"""
        <a href="#card-{html.escape(uc_id)}" class="nav-link">
            <span>{html.escape(uc_id)}</span>
            <span class="status-dot {badge_dot_class}"></span>
        </a>
        """

        error_panel = ""
        if uc.get("error"):
            error_label = "Study Finding" if uc.get("status") in {"known_limit", "timeout"} else "Execution Error"
            error_panel = f"""
            <div class="error-box">
                <strong>{html.escape(error_label)}:</strong>
                <pre>{html.escape(uc.get("error", ""))}</pre>
            </div>
            """

        metric_rows = ""
        for key, value in metrics.items():
            metric_rows += f"<li><strong>{html.escape(str(key))}:</strong> {html.escape(str(value))}</li>"

        test_case_rows = ""
        for tc_idx, tc in enumerate(test_cases, 1):
            input_val = str(tc.get("input", ""))
            truncated_input = input_val if len(input_val) <= 200 else input_val[:197] + "..."
            expected_val = json.dumps(tc.get("expected", ""), ensure_ascii=False)
            actual_val = (
                json.dumps(tc.get("actual", ""), ensure_ascii=False)
                if tc.get("actual") is not None
                else "null"
            )
            passed = tc.get("passed", False)
            test_case_rows += f"""
            <tr>
                <td>{tc_idx}</td>
                <td title="{html.escape(input_val)}">{html.escape(truncated_input)}</td>
                <td class="mono">{html.escape(expected_val)}</td>
                <td class="mono">{html.escape(actual_val)}</td>
                <td><span class="status-tag {'pass-tag' if passed else 'fail-tag'}">{'PASS' if passed else 'FAIL'}</span></td>
                <td>{tc.get('inference_time_s', 0.0):.4f}s</td>
                <td class="notes-cell">{html.escape(str(tc.get('notes', '')))}</td>
            </tr>
            """

        use_case_details_html += f"""
        <div class="use-case-card" id="card-{html.escape(uc_id)}">
            <div class="use-case-header" onclick="toggleCard('{html.escape(uc_id)}')">
                <div class="header-left">
                    <span class="collapse-icon" id="icon-{html.escape(uc_id)}">&#9660;</span>
                    <h3>{html.escape(uc_id)}</h3>
                    <span class="type-badge-inline">{html.escape(problem_name)}</span>
                </div>
                <span class="status-badge {badge_class}">{badge_text}</span>
            </div>
            <div class="use-case-content" id="content-{html.escape(uc_id)}">
                <div class="metadata-grid">
                    <div>
                        <strong>Problem</strong>
                        <p>{html.escape(problem_name)}</p>
                    </div>
                    <div>
                        <strong>Technique</strong>
                        <p>{html.escape(technique_name)}</p>
                    </div>
                    <div>
                        <strong>Application</strong>
                        <p>{html.escape(application_name)}</p>
                    </div>
                    <div>
                        <strong>Model &amp; Library</strong>
                        <p class="mono">{html.escape(model)} ({html.escape(uc.get('library', 'unknown'))})</p>
                        <div class="code-info-container">
                            <span class="info-icon-badge">View Code &amp; Params</span>
                            <div class="code-tooltip">
                                <strong>Loader</strong>
                                <pre>{html.escape(code_details["loader"])}</pre>
                                <strong>Parameters</strong>
                                <pre>{html.escape(code_details["params"])}</pre>
                                <strong>Inference Call</strong>
                                <pre>{html.escape(code_details["call"])}</pre>
                            </div>
                        </div>
                    </div>
                    <div>
                        <strong>Runtime Tier</strong>
                        <p>{html.escape(runtime_tier)}</p>
                    </div>
                    <div>
                        <strong>Primary Metric</strong>
                        <p>{html.escape(primary_metric)}</p>
                    </div>
                    <div>
                        <strong>Performance</strong>
                        <p>Load: {load_time:.3f}s | Inference: {uc.get('total_inference_time_s', 0.0):.3f}s | Total: {uc.get('total_runtime_s', 0.0):.3f}s</p>
                    </div>
                    <div>
                        <strong>Completed At</strong>
                        <p>{html.escape(completed_at)}</p>
                    </div>
                </div>
                <div class="metrics-box">
                    <strong>Metrics</strong>
                    <ul>{metric_rows}</ul>
                </div>
                {error_panel}
                <h4>Evaluation Examples</h4>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 5%">#</th>
                                <th style="width: 35%">Input</th>
                                <th style="width: 15%">Expected</th>
                                <th style="width: 15%">Actual</th>
                                <th style="width: 10%">Status</th>
                                <th style="width: 10%">Time</th>
                                <th style="width: 10%">Notes</th>
                            </tr>
                        </thead>
                        <tbody>
                            {test_case_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLP Offline Study Report</title>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --text-color: #1e293b;
            --card-bg: #ffffff;
            --border-color: #e2e8f0;
            --primary: #4f46e5;
            --success: #16a34a;
            --danger: #dc2626;
            --amber: #d97706;
            --muted: #64748b;
        }}

        body {{
            font-family: "Segoe UI", system-ui, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            line-height: 1.5;
        }}

        .app-layout {{
            display: flex;
            min-height: 100vh;
        }}

        .sidebar {{
            width: 280px;
            background-color: #0f172a;
            color: #f8fafc;
            position: fixed;
            top: 0;
            bottom: 0;
            left: 0;
            padding: 1.5rem 1rem;
            box-sizing: border-box;
            overflow-y: auto;
        }}

        .sidebar-header h2 {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: #94a3b8;
            margin: 0 0 1.25rem 0.75rem;
        }}

        .sidebar-nav {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .nav-link {{
            color: #94a3b8;
            text-decoration: none;
            font-size: 0.825rem;
            padding: 0.5rem 0.75rem;
            border-radius: 0.375rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .nav-link:hover,
        .nav-link.active {{
            background-color: var(--primary);
            color: #ffffff;
        }}

        .nav-section-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #475569;
            margin: 1.25rem 0 0.5rem 0.75rem;
            font-weight: 700;
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }}

        .main-content {{
            flex: 1;
            margin-left: 280px;
            padding: 2.5rem 2rem;
            box-sizing: border-box;
        }}

        .container {{
            max-width: 1080px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 2rem;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 1.5rem;
        }}

        h1 {{
            margin: 0 0 0.5rem 0;
            font-size: 2rem;
            font-weight: 700;
        }}

        .meta-info {{
            font-size: 0.85rem;
            color: var(--muted);
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .summary-card,
        .card,
        .use-case-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}

        .summary-card {{
            padding: 1.25rem;
        }}

        .summary-card .label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--muted);
            font-weight: 600;
        }}

        .summary-card .value {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }}

        .card {{
            overflow: visible;
            margin-bottom: 2rem;
        }}

        .card-header {{
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--border-color);
            background-color: #f1f5f9;
        }}

        .table-container {{
            width: 100%;
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
            table-layout: fixed;
        }}

        th,
        td {{
            padding: 0.7rem 0.55rem;
            border-bottom: 1px solid var(--border-color);
            text-align: left;
            vertical-align: top;
            overflow-wrap: break-word;
        }}

        th {{
            background-color: #f8fafc;
            color: var(--muted);
            text-transform: uppercase;
            font-size: 0.725rem;
            letter-spacing: 0.05em;
        }}

        .use-cases-list {{
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }}

        .use-case-card {{
            overflow: visible;
            scroll-margin-top: 1rem;
        }}

        .use-case-header {{
            padding: 1rem 1.25rem;
            background-color: #f8fafc;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }}

        .use-case-content {{
            padding: 1.25rem;
            border-top: 1px solid var(--border-color);
            overflow: visible;
        }}

        .header-left {{
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }}

        .collapse-icon {{
            color: var(--muted);
        }}

        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 1.25rem;
        }}

        .metadata-grid p {{
            margin: 0.25rem 0 0 0;
            color: #475569;
            font-size: 0.85rem;
        }}

        .metrics-box {{
            background-color: #f8fafc;
            border: 1px solid var(--border-color);
            border-radius: 0.375rem;
            padding: 0.85rem 1rem;
            margin-bottom: 1.25rem;
        }}

        .metrics-box ul {{
            margin: 0.5rem 0 0 1rem;
            padding: 0;
        }}

        .code-info-container {{
            position: relative;
            display: inline-block;
            margin-top: 0.55rem;
        }}

        .info-icon-badge {{
            background-color: #eef2ff;
            color: #4338ca;
            border: 1px dashed #c7d2fe;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.18rem 0.45rem;
            border-radius: 0.25rem;
            cursor: default;
        }}

        .code-tooltip {{
            display: none;
            position: absolute;
            top: 125%;
            left: 0;
            max-width: min(420px, calc(100vw - 3rem));
            background-color: #0f172a;
            color: #e2e8f0;
            border: 1px solid #334155;
            border-radius: 0.5rem;
            padding: 0.8rem 0.9rem;
            width: 360px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.25);
            z-index: 20;
        }}

        .code-info-container:hover .code-tooltip {{
            display: block;
        }}

        .code-tooltip strong {{
            display: block;
            margin-top: 0.4rem;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #cbd5e1;
        }}

        .code-tooltip strong:first-child {{
            margin-top: 0;
        }}

        .code-tooltip pre {{
            white-space: pre-wrap;
            word-break: break-word;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Courier New", monospace;
            font-size: 0.72rem;
            margin: 0.25rem 0 0 0;
            color: #f8fafc;
        }}

        .type-badge-inline {{
            background-color: #e0e7ff;
            color: #4338ca;
            font-size: 0.725rem;
            padding: 0.2rem 0.45rem;
            border-radius: 0.25rem;
            font-weight: 500;
        }}

        .status-badge {{
            font-size: 0.725rem;
            font-weight: 700;
            padding: 0.3rem 0.65rem;
            border-radius: 9999px;
            letter-spacing: 0.05em;
        }}

        .badge-pass {{
            background-color: #dcfce7;
            color: #15803d;
        }}

        .badge-amber {{
            background-color: #fef3c7;
            color: #b45309;
        }}

        .badge-fail {{
            background-color: #fee2e2;
            color: #b91c1c;
        }}

        .status-tag {{
            font-size: 0.725rem;
            font-weight: 600;
            padding: 0.15rem 0.45rem;
            border-radius: 0.25rem;
        }}

        .pass-tag {{
            background-color: #dcfce7;
            color: #16a34a;
        }}

        .fail-tag {{
            background-color: #fee2e2;
            color: #dc2626;
        }}

        .pass-text {{
            color: var(--success);
        }}

        .amber-text {{
            color: var(--amber);
        }}

        .fail-text {{
            color: var(--danger);
        }}

        .font-bold {{
            font-weight: 700;
        }}

        .mono {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Courier New", monospace;
            font-size: 0.75rem;
        }}

        .col-small {{
            width: 3.5%;
        }}

        .col-use-case {{
            width: 12%;
        }}

        .col-problem {{
            width: 11%;
        }}

        .col-technique {{
            width: 12%;
        }}

        .col-tier {{
            width: 5%;
        }}

        .col-model {{
            width: 31%;
            overflow-wrap: anywhere;
        }}

        .col-metric {{
            width: 6.5%;
            white-space: nowrap;
        }}

        .notes-cell {{
            font-size: 0.775rem;
            color: var(--muted);
        }}

        .error-box {{
            background-color: #fef2f2;
            border: 1px solid #fee2e2;
            border-left: 4px solid var(--danger);
            padding: 1rem;
            border-radius: 0.375rem;
            margin-bottom: 1rem;
            color: #991b1b;
        }}

        .error-box pre {{
            white-space: pre-wrap;
            word-break: break-word;
            margin: 0.5rem 0 0 0;
        }}

        footer {{
            color: var(--muted);
            font-size: 0.85rem;
            margin: 2rem 0 1rem 0;
        }}

        @media (max-width: 960px) {{
            .sidebar {{
                position: static;
                width: 100%;
            }}

            .main-content {{
                margin-left: 0;
                padding: 1.5rem 1rem;
            }}

            .app-layout {{
                display: block;
            }}
        }}
    </style>
    <script>
        function toggleCard(ucId) {{
            const content = document.getElementById(`content-${{ucId}}`);
            const icon = document.getElementById(`icon-${{ucId}}`);
            if (!content || !icon) return;
            const isHidden = content.style.display === "none";
            content.style.display = isHidden ? "block" : "none";
            icon.innerHTML = isHidden ? "&#9660;" : "&#9658;";
        }}

        document.addEventListener("DOMContentLoaded", () => {{
            document.querySelectorAll(".nav-link").forEach((link) => {{
                link.addEventListener("click", (event) => {{
                    event.preventDefault();
                    const target = document.querySelector(link.getAttribute("href"));
                    if (!target) return;
                    target.scrollIntoView({{ behavior: "smooth" }});
                    document.querySelectorAll(".nav-link").forEach((node) => node.classList.remove("active"));
                    link.classList.add("active");
                }});
            }});
        }});
    </script>
</head>
<body>
    <div class="app-layout">
        <aside class="sidebar">
            <div class="sidebar-header">
                <h2>Offline Study</h2>
            </div>
            <nav class="sidebar-nav">
                <a href="#header-section" class="nav-link active"><span>Overview</span></a>
                <a href="#summary-section" class="nav-link"><span>Summary Metrics</span></a>
                <a href="#comparison-section" class="nav-link"><span>Comparison Table</span></a>
                <div class="nav-section-title">Studies</div>
                {nav_links_html}
            </nav>
        </aside>
        <main class="main-content">
            <div class="container">
                <section id="header-section">
                    <header>
                        <h1>Offline NLP Study Report</h1>
                        <div class="meta-info">
                            <span>Started: <strong>{html.escape(run_metadata.get("started_at", "N/A"))}</strong></span>
                            <span>Finished: <strong>{html.escape(run_metadata.get("finished_at", "N/A"))}</strong></span>
                            <span>Total Wall Time: <strong>{run_metadata.get("total_wall_time_s", 0.0):.2f}s</strong></span>
                            <span>Host OS: <strong>{html.escape(run_metadata.get("host_os", "N/A"))}</strong></span>
                            <span>Python: <strong>{html.escape(run_metadata.get("python_version", "N/A"))}</strong></span>
                        </div>
                    </header>
                </section>
                <section id="summary-section">
                    <div class="summary-grid">
                        <div class="summary-card">
                            <div class="label">Total Studies</div>
                            <div class="value">{summary.get("total_use_cases", 0)}</div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Successful Runs</div>
                            <div class="value">{summary.get("successful_use_cases", 0)}</div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Total Examples</div>
                            <div class="value">{summary.get("total_test_cases", 0)}</div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Overall Pass Rate</div>
                            <div class="value">{overall_pass_rate_pct}</div>
                        </div>
                    </div>
                </section>
                <section id="comparison-section">
                    <div class="card">
                        <div class="card-header">
                            <h2>Problem / Technique Comparison</h2>
                        </div>
                        <div class="table-container">
                            <table>
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Use Case</th>
                                        <th>Problem</th>
                                        <th>Technique</th>
                                        <th>Tier</th>
                                        <th>Model</th>
                                        <th>Load</th>
                                        <th>Avg Inference</th>
                                        <th>Pass Rate</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {comparison_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </section>
                <section id="details-section">
                    <h2>Detailed Results</h2>
                    <div class="use-cases-list">
                        {use_case_details_html}
                    </div>
                </section>
                <footer>
                    Generated by <code>nlp-without-llm</code> using local study artifacts.
                </footer>
            </div>
        </main>
    </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
