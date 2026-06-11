import json
import html
from pathlib import Path

# Definition of code snippets and parameters used for each use case
CODE_DETAILS = {
    "01_language_detection": {
        "loader": "pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')",
        "params": "None",
        "call": "classifier(text)"
    },
    "02_translate_zh_en": {
        "loader": "AutoTokenizer.from_pretrained(model, src_lang='zho_Hans')\nAutoModelForSeq2SeqLM.from_pretrained(model)\nSentenceTransformer('all-mpnet-base-v2', device='cpu')",
        "params": "forced_bos_token_id = tokenizer.convert_tokens_to_ids('eng_Latn')",
        "call": "model.generate(**inputs, forced_bos_token_id=...)\n# Evaluate cosine similarity against reference using MPNet"
    },
    "03_sentiment_customer_review": {
        "loader": "pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')",
        "params": "None",
        "call": "classifier(text)"
    },
    "04_sentiment_internal_escalation": {
        "loader": "pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')",
        "params": "None",
        "call": "classifier(text)\n# Map negative to 'escalated', positive/neutral to 'not_escalated'"
    },
    "05_ner_brands_locations": {
        "loader": "pipeline('ner', model='dslim/bert-base-NER', aggregation_strategy='simple')",
        "params": "None",
        "call": "ner(text)\n# Evaluate set-overlap F1 score (threshold >= 0.66)"
    },
    "06_ner_people_orgs_procurement": {
        "loader": "pipeline('ner', model='dslim/bert-base-NER', aggregation_strategy='simple')",
        "params": "None",
        "call": "ner(text)\n# Evaluate set-overlap F1 score (threshold >= 0.66)"
    },
    "07_classify_email_department": {
        "loader": "pipeline('zero-shot-classification', model='MoritzLaurer/deberta-v3-base-zeroshot-v2.0')",
        "params": "candidate_labels=['IT', 'Planner', 'Pricing', 'Warehouse', 'Finance', 'HR']",
        "call": "classifier(text, candidate_labels=CANDIDATE_LABELS)"
    },
    "08_classify_supplier_urgency": {
        "loader": "pipeline('zero-shot-classification', model='MoritzLaurer/deberta-v3-base-zeroshot-v2.0')",
        "params": "candidate_labels=['urgent', 'routine', 'dispute', 'delivery_update', 'invoice_query']",
        "call": "classifier(text, candidate_labels=CANDIDATE_LABELS)"
    },
    "09_classify_feedback_topic": {
        "loader": "pipeline('zero-shot-classification', model='MoritzLaurer/deberta-v3-base-zeroshot-v2.0')",
        "params": "candidate_labels=['product_quality', 'staff_service', 'pricing', 'store_experience', 'checkout_process']",
        "call": "classifier(text, candidate_labels=CANDIDATE_LABELS)"
    },
    "10_similarity_complaint_lookup": {
        "loader": "SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')",
        "params": "corpus_embeddings = model.encode(corpus_texts)",
        "call": "query_embedding = model.encode(query)\nutil.cos_sim(query_embedding, corpus_embeddings)[0]"
    },
    "11_similarity_sop_retrieval": {
        "loader": "SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')",
        "params": "corpus_embeddings = model.encode(corpus_texts)",
        "call": "query_embedding = model.encode(query)\nutil.cos_sim(query_embedding, corpus_embeddings)[0]"
    },
    "12_fewshot_email_department": {
        "loader": "AutoTokenizer.from_pretrained(model)\nAutoModelForSequenceClassification.from_pretrained(model)",
        "params": "FEW_SHOT_CONTEXT = '...[10 email classification examples]...'",
        "call": "premise = FEW_SHOT_CONTEXT.format(text=text)\n# Evaluate custom hypotheses using NLI"
    },
    "13_fewshot_supplier_urgency": {
        "loader": "AutoTokenizer.from_pretrained(model)\nAutoModelForSequenceClassification.from_pretrained(model)",
        "params": "FEW_SHOT_CONTEXT = '...[10 supplier email examples]...'",
        "call": "premise = FEW_SHOT_CONTEXT.format(text=text)\n# Evaluate custom hypotheses using NLI"
    },
    "14_fewshot_feedback_topic": {
        "loader": "AutoTokenizer.from_pretrained(model)\nAutoModelForSequenceClassification.from_pretrained(model)",
        "params": "FEW_SHOT_CONTEXT = '...[10 customer feedback examples]...'",
        "call": "premise = FEW_SHOT_CONTEXT.format(text=text)\n# Evaluate custom hypotheses using NLI"
    }
}

def render(results: dict, output_path: Path = Path("output/report.html")) -> None:
    """Renders a self-contained HTML report from the aggregated results."""
    run_metadata = results.get("run_metadata", {})
    summary = results.get("summary", {})
    use_cases = results.get("use_cases", [])

    # Calculate overall pass rate percentage
    overall_pass_rate_pct = f"{round(summary.get('overall_pass_rate', 0.0) * 100, 2)}%"

    # Generate model comparison table rows
    comparison_rows = ""
    for idx, uc in enumerate(use_cases, 1):
        uc_id = uc.get("use_case_id", "")
        uc_type = uc.get("type", "")
        model = uc.get("model", "")
        load_time = uc.get("model_load_time_s", 0.0)
        
        test_cases = uc.get("test_cases", [])
        if test_cases:
            avg_inference = sum(tc.get("inference_time_s", 0.0) for tc in test_cases) / len(test_cases)
        else:
            avg_inference = 0.0
            
        pass_rate_val = uc.get("pass_rate", 0.0)
        pass_rate_str = f"{round(pass_rate_val * 100, 1)}%"
        
        # Color coding for pass rate cell in table
        if pass_rate_val >= 0.66:
            pass_rate_class = "pass-text"
        elif pass_rate_val >= 0.33:
            pass_rate_class = "amber-text"
        else:
            pass_rate_class = "fail-text"

        comparison_rows += f"""
        <tr>
            <td>{idx}</td>
            <td><strong>{html.escape(uc_id)}</strong></td>
            <td class="badge-cell"><span class="type-badge">{html.escape(uc_type)}</span></td>
            <td class="mono">{html.escape(model)}</td>
            <td>{load_time:.3f}s</td>
            <td>{avg_inference:.4f}s</td>
            <td class="{pass_rate_class} font-bold">{pass_rate_str}</td>
        </tr>
        """

    # Generate per-use-case details and sidebar nav links
    use_case_details_html = ""
    nav_links_html = ""
    
    for uc in use_cases:
        uc_id = uc.get("use_case_id", "")
        code_details = CODE_DETAILS.get(uc_id, {
            "loader": "N/A",
            "params": "N/A",
            "call": "N/A"
        })
        uc_type = uc.get("type", "")
        description = uc.get("description", "")
        relevance = uc.get("domain_relevance", "")
        model = uc.get("model", "")
        library = uc.get("library", "")
        load_time = uc.get("model_load_time_s", 0.0)
        total_inference = uc.get("total_inference_time_s", 0.0)
        total_runtime = uc.get("total_runtime_s", 0.0)
        error = uc.get("error")
        status = uc.get("status", "ok" if error is None else "failed")
        pass_rate_val = uc.get("pass_rate", 0.0)
        pass_count = uc.get("pass_count", 0)
        total_count = uc.get("total_count", 0)

        # Style badges based on pass rate
        if status == "known_limit":
            badge_class = "badge-fail"
            badge_text = "BETTER VIA API"
            badge_dot_class = "badge-fail"
        elif status == "timeout":
            badge_class = "badge-fail"
            badge_text = "TIMEOUT"
            badge_dot_class = "badge-fail"
        elif error:
            badge_class = "badge-fail"
            badge_text = "ERROR"
            badge_dot_class = "badge-fail"
        elif pass_rate_val >= 0.66:
            badge_class = "badge-pass"
            badge_text = f"PASS ({pass_count}/{total_count})"
            badge_dot_class = "badge-pass"
        elif pass_rate_val >= 0.33:
            badge_class = "badge-amber"
            badge_text = f"BORDERLINE ({pass_count}/{total_count})"
            badge_dot_class = "badge-amber"
        else:
            badge_class = "badge-fail"
            badge_text = f"FAIL ({pass_count}/{total_count})"
            badge_dot_class = "badge-fail"

        # Generate Sidebar Navigation Link
        nav_links_html += f"""
        <a href="#card-{html.escape(uc_id)}" class="nav-link">
            <span>{html.escape(uc_id)}</span>
            <span class="status-dot {badge_dot_class}"></span>
        </a>
        """

        # Error panel if the script crashed
        error_panel = ""
        if error:
            error_label = "Study Finding" if status in {"known_limit", "timeout"} else "Execution Error"
            error_panel = f"""
            <div class="error-box">
                <strong>{html.escape(error_label)}:</strong>
                <pre>{html.escape(error)}</pre>
            </div>
            """

        # Build test cases table
        test_case_rows = ""
        for tc_idx, tc in enumerate(uc.get("test_cases", []), 1):
            input_val = str(tc.get("input", ""))
            # Truncate input if too long but keep full in title for hover
            truncated_input = input_val if len(input_val) <= 200 else input_val[:197] + "..."
            
            expected_val = json.dumps(tc.get("expected", ""), ensure_ascii=False)
            actual_val = json.dumps(tc.get("actual", ""), ensure_ascii=False) if tc.get("actual") is not None else "null"
            
            passed = tc.get("passed", False)
            status_text = "PASS" if passed else "FAIL"
            status_class = "pass-tag" if passed else "fail-tag"
            
            inference_time = tc.get("inference_time_s", 0.0)
            notes = tc.get("notes", "")

            test_case_rows += f"""
            <tr>
                <td>{tc_idx}</td>
                <td title="{html.escape(input_val)}">{html.escape(truncated_input)}</td>
                <td class="mono">{html.escape(expected_val)}</td>
                <td class="mono">{html.escape(actual_val)}</td>
                <td><span class="status-tag {status_class}">{status_text}</span></td>
                <td>{inference_time:.4f}s</td>
                <td class="notes-cell">{html.escape(notes)}</td>
            </tr>
            """

        use_case_details_html += f"""
        <div class="use-case-card" id="card-{html.escape(uc_id)}">
            <div class="use-case-header" onclick="toggleCard('{html.escape(uc_id)}')">
                <div class="header-left">
                    <span class="collapse-icon" id="icon-{html.escape(uc_id)}">▼</span>
                    <h3>{html.escape(uc_id)}</h3>
                    <span class="type-badge-inline">{html.escape(uc_type)}</span>
                </div>
                <span class="status-badge {badge_class}">{badge_text}</span>
            </div>
            
            <div class="use-case-content" id="content-{html.escape(uc_id)}">
                <div class="metadata-grid">
                    <div>
                        <strong>Description:</strong>
                        <p>{html.escape(description)}</p>
                    </div>
                    <div>
                        <strong>Domain Relevance:</strong>
                        <p>{html.escape(relevance)}</p>
                    </div>
                    <div>
                        <strong>Model &amp; Library:</strong>
                        <p class="mono">{html.escape(model)} ({html.escape(library)})</p>
                        <div style="margin-top: 0.5rem;">
                            <div class="code-info-container">
                                <span class="info-icon-badge">ℹ️ View Code &amp; Params</span>
                                <div class="code-tooltip">
                                    <strong>Loader Snippet:</strong>
                                    <pre>{html.escape(code_details.get("loader", "N/A"))}</pre>
                                    <strong>Parameters:</strong>
                                    <pre>{html.escape(code_details.get("params", "N/A"))}</pre>
                                    <strong>Inference Call:</strong>
                                    <pre>{html.escape(code_details.get("call", "N/A"))}</pre>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div>
                        <strong>Performance Summary:</strong>
                        <p>Load: {load_time:.3f}s | Inference: {total_inference:.3f}s | Total: {total_runtime:.3f}s</p>
                    </div>
                </div>

                {error_panel}

                <h4>Test Cases</h4>
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

    # Assemble full HTML file content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLP Offline Evaluation Lab Report</title>
    <style>
        /* Base styles */
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
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            line-height: 1.5;
            display: flex;
        }}

        /* App Layout structure */
        .app-layout {{
            display: flex;
            width: 100%;
            min-height: 100vh;
        }}

        /* Sidebar Navigation */
        .sidebar {{
            width: 300px;
            background-color: #0f172a;
            color: #f8fafc;
            position: fixed;
            top: 0;
            bottom: 0;
            left: 0;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #1e293b;
            padding: 1.5rem 1rem;
            box-sizing: border-box;
            overflow-y: auto;
        }}

        .sidebar-header h2 {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--muted);
            margin: 0 0 1.5rem 0.75rem;
            font-weight: 700;
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
            transition: all 0.2s;
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .nav-link:hover {{
            background-color: #1e293b;
            color: #f8fafc;
        }}

        .nav-link.active {{
            background-color: var(--primary);
            color: #ffffff;
            font-weight: 600;
        }}

        .nav-section-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #475569;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            margin-left: 0.75rem;
            font-weight: 700;
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            flex-shrink: 0;
        }}

        /* Main Content Container */
        .main-content {{
            flex: 1;
            margin-left: 300px;
            padding: 2.5rem 2rem;
            box-sizing: border-box;
        }}

        .container {{
            max-width: 1000px;
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
            color: #0f172a;
            letter-spacing: -0.025em;
        }}

        .meta-info {{
            font-size: 0.85rem;
            color: var(--muted);
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
        }}

        .meta-info span strong {{
            color: var(--text-color);
        }}

        /* Disclosure Box */
        .disclosure-box {{
            background-color: #eff6ff;
            border-left: 4px solid var(--primary);
            color: #1e40af;
            padding: 1rem;
            border-radius: 0.375rem;
            margin-bottom: 2rem;
            font-size: 0.9rem;
            font-weight: 500;
        }}

        /* Summary Cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }}

        .summary-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}

        .summary-card .label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--muted);
            font-weight: 600;
        }}

        .summary-card .value {{
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.25rem;
            color: #0f172a;
        }}

        /* Comparison Table */
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            margin-bottom: 2.5rem;
            overflow: hidden;
        }}

        .card-header {{
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            background-color: #f1f5f9;
        }}

        .card-header h2 {{
            margin: 0;
            font-size: 1.15rem;
            font-weight: 600;
            color: #0f172a;
        }}

        .table-container {{
            width: 100%;
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.85rem;
        }}

        th {{
            background-color: #f8fafc;
            color: var(--muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.725rem;
            letter-spacing: 0.05em;
            padding: 0.75rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }}

        td {{
            padding: 0.85rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            vertical-align: middle;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        /* Use Case Cards list */
        .use-cases-list {{
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }}

        .use-case-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            overflow: visible; /* Required for hover tooltips to show correctly */
            scroll-margin-top: 2rem; /* Add spacing when clicking navigation links */
        }}

        .use-case-header {{
            padding: 1.25rem 1.5rem;
            background-color: #f8fafc;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
            transition: background-color 0.2s;
        }}

        .use-case-header:hover {{
            background-color: #f1f5f9;
        }}

        .header-left {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .collapse-icon {{
            font-size: 0.75rem;
            color: var(--muted);
            transition: transform 0.2s;
            display: inline-block;
        }}

        .use-case-header h3 {{
            margin: 0;
            font-size: 1.05rem;
            font-weight: 600;
            color: #0f172a;
        }}

        .use-case-content {{
            padding: 1.5rem;
            border-top: 1px solid var(--border-color);
        }}

        /* Grid */
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
            margin-bottom: 1.5rem;
        }}

        .metadata-grid p {{
            margin: 0.25rem 0 0 0;
            color: #475569;
            font-size: 0.85rem;
        }}

        /* Badges and Tags */
        .type-badge {{
            background-color: #e2e8f0;
            color: #475569;
            font-size: 0.725rem;
            padding: 0.2rem 0.45rem;
            border-radius: 0.25rem;
            font-weight: 500;
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

        /* Text colors */
        .pass-text {{
            color: var(--success);
        }}

        .fail-text {{
            color: var(--danger);
        }}

        .amber-text {{
            color: var(--amber);
        }}

        .font-bold {{
            font-weight: 700;
        }}

        /* Code and Typography helpers */
        .mono {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 0.8rem;
            color: #0f172a;
        }}

        .error-box {{
            background-color: #fef2f2;
            border: 1px solid #fee2e2;
            border-left: 4px solid var(--danger);
            padding: 1rem;
            border-radius: 0.375rem;
            margin-bottom: 1.5rem;
            color: #991b1b;
        }}

        .error-box pre {{
            margin: 0.5rem 0 0 0;
            font-family: ui-monospace, monospace;
            font-size: 0.775rem;
            white-space: pre-wrap;
            word-break: break-all;
        }}

        .notes-cell {{
            font-size: 0.775rem;
            color: var(--muted);
        }}

        /* Code reference tooltip */
        .code-info-container {{
            position: relative;
            display: inline-block;
            cursor: pointer;
        }}

        .info-icon-badge {{
            background-color: #f0f9ff;
            color: #0284c7;
            border: 1px dashed #bae6fd;
            font-size: 0.725rem;
            font-weight: 600;
            padding: 0.15rem 0.45rem;
            border-radius: 0.25rem;
            transition: all 0.2s;
            user-select: none;
        }}

        .code-info-container:hover .info-icon-badge {{
            background-color: #e0f2fe;
            border-color: #7dd3fc;
        }}

        .code-tooltip {{
            display: none;
            position: absolute;
            top: 125%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #0f172a;
            color: #e2e8f0;
            border: 1px solid #334155;
            border-radius: 0.5rem;
            padding: 0.75rem 1rem;
            width: 320px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.3);
            z-index: 2000;
            text-align: left;
        }}

        .code-tooltip::after {{
            content: "";
            position: absolute;
            bottom: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: transparent transparent #0f172a transparent;
        }}

        .code-info-container:hover .code-tooltip {{
            display: block;
        }}

        .code-tooltip strong {{
            display: block;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-top: 0.5rem;
            margin-bottom: 0.25rem;
        }}

        .code-tooltip strong:first-of-type {{
            margin-top: 0;
        }}

        .code-tooltip pre {{
            margin: 0;
            background-color: #1e293b;
            padding: 0.4rem;
            border-radius: 0.25rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.725rem;
            white-space: pre-wrap;
            word-break: break-all;
            color: #38bdf8;
        }}

        footer {{
            margin-top: 4rem;
            text-align: center;
            font-size: 0.85rem;
            color: var(--muted);
            border-top: 1px solid var(--border-color);
            padding-top: 2rem;
        }}

        /* JS functionality styling */
        .use-case-content.collapsed {{
            display: none;
        }}

        .collapse-icon.collapsed {{
            transform: rotate(-90deg);
        }}

        /* Responsive styling for smaller screens */
        @media (max-width: 900px) {{
            body {{
                display: block;
            }}
            .sidebar {{
                position: relative;
                width: 100%;
                height: auto;
                border-right: none;
                border-bottom: 1px solid #1e293b;
            }}
            .main-content {{
                margin-left: 0;
                padding: 1.5rem 1rem;
            }}
        }}
    </style>
    <script>
        function toggleCard(ucId) {{
            const content = document.getElementById('content-' + ucId);
            const icon = document.getElementById('icon-' + ucId);
            if (content.classList.contains('collapsed')) {{
                content.classList.remove('collapsed');
                icon.classList.remove('collapsed');
            }} else {{
                content.classList.add('collapsed');
                icon.classList.add('collapsed');
            }}
        }}

        // Automatic ScrollSpy implementation using IntersectionObserver
        document.addEventListener('DOMContentLoaded', () => {{
            const observerOptions = {{
                root: null,
                threshold: 0.15,
                rootMargin: '0px 0px -50% 0px'
            }};

            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting) {{
                        const id = entry.target.getAttribute('id');
                        let navLink;
                        
                        if (id === 'header-section') {{
                            navLink = document.querySelector('.nav-link[href="#header-section"]');
                        }} else if (id === 'summary-section') {{
                            navLink = document.querySelector('.nav-link[href="#summary-section"]');
                        }} else if (id === 'comparison-section') {{
                            navLink = document.querySelector('.nav-link[href="#comparison-section"]');
                        }} else {{
                            navLink = document.querySelector(`.nav-link[href="#${{id}}"]`);
                        }}

                        if (navLink) {{
                            document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
                            navLink.classList.add('active');
                        }}
                    }}
                }});
            }}, observerOptions);

            // Observe sections
            document.querySelectorAll('section[id], .use-case-card[id]').forEach((section) => {{
                observer.observe(section);
            }});

            // Smooth scrolling behavior on click
            document.querySelectorAll('.nav-link').forEach(link => {{
                link.addEventListener('click', (e) => {{
                    e.preventDefault();
                    const targetId = link.getAttribute('href');
                    const targetElement = document.querySelector(targetId);
                    if (targetElement) {{
                        targetElement.scrollIntoView({{ behavior: 'smooth' }});
                        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                        link.classList.add('active');
                    }}
                }});
            }});
        }});
    </script>
</head>
<body>
    <div class="app-layout">
        <!-- Sidebar Navigation -->
        <aside class="sidebar">
            <div class="sidebar-header">
                <h2>Evaluation Lab</h2>
            </div>
            <nav class="sidebar-nav">
                <a href="#header-section" class="nav-link active">
                    <span>Overview</span>
                </a>
                <a href="#summary-section" class="nav-link">
                    <span>Summary Metrics</span>
                </a>
                <a href="#comparison-section" class="nav-link">
                    <span>Model Comparison</span>
                </a>
                <div class="nav-section-title">Use Cases</div>
                {nav_links_html}
            </nav>
        </aside>

        <!-- Main Content Area -->
        <main class="main-content">
            <div class="container">
                <section id="header-section">
                    <header>
                        <h1>NLP Offline Evaluation Lab Report</h1>
                        <div class="meta-info">
                            <span>Started: <strong>{html.escape(run_metadata.get("started_at", "N/A"))}</strong></span>
                            <span>Finished: <strong>{html.escape(run_metadata.get("finished_at", "N/A"))}</strong></span>
                            <span>Total Wall Time: <strong>{run_metadata.get("total_wall_time_s", 0.0):.2f}s</strong></span>
                            <span>Host OS: <strong>{html.escape(run_metadata.get("host_os", "N/A"))}</strong></span>
                            <span>Python: <strong>{html.escape(run_metadata.get("python_version", "N/A"))}</strong></span>
                        </div>
                    </header>

                    <div class="disclosure-box">
                        Pass rates reflect zero-shot performance with no fine-tuning. Honest baseline on a tiny test set (3-5 cases per use case) — not a benchmark.
                    </div>
                </section>

                <section id="summary-section">
                    <div class="summary-grid">
                        <div class="summary-card">
                            <div class="label">Total Use Cases</div>
                            <div class="value">{summary.get("total_use_cases", 0)}</div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Successful Runs</div>
                            <div class="value">{summary.get("successful_use_cases", 0)}</div>
                        </div>
                        <div class="summary-card">
                            <div class="label">Total Test Cases</div>
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
                            <h2>Model Comparison Summary</h2>
                        </div>
                        <div class="table-container">
                            <table>
                                <thead>
                                    <tr>
                                        <th style="width: 5%">#</th>
                                        <th style="width: 25%">Use Case</th>
                                        <th style="width: 15%">Type</th>
                                        <th style="width: 25%">Model</th>
                                        <th style="width: 10%">Load Time</th>
                                        <th style="width: 10%">Avg Inference</th>
                                        <th style="width: 10%">Pass Rate</th>
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
                    <h2>Detailed Results Per Use Case</h2>
                    <div class="use-cases-list">
                        {use_case_details_html}
                    </div>
                </section>

                <footer>
                    Generated by <code>nlp-without-llm</code> — pretrained models, no LLM API calls.
                </footer>
            </div>
        </main>
    </div>
</body>
</html>
"""

    # Write output to HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML report successfully rendered to {output_path}")
