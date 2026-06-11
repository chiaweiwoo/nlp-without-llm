import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


def load_json_fixture(relative_path: str) -> dict | list:
    fixture_path = DATA_DIR / relative_path
    with open(fixture_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_sentiment_label(label: str) -> str:
    lowered = label.lower()
    if "positive" in lowered or lowered in {"label_2", "pos"}:
        return "positive"
    if "negative" in lowered or lowered in {"label_0", "neg"}:
        return "negative"
    if "neutral" in lowered or lowered in {"label_1", "neu"}:
        return "neutral"
    return lowered


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def lexicon_sentiment(
    text: str,
    positive_terms: set[str],
    negative_terms: set[str],
) -> tuple[str, dict]:
    tokens = tokenize_words(text)
    positive_hits = sum(1 for token in tokens if token in positive_terms)
    negative_hits = sum(1 for token in tokens if token in negative_terms)

    if positive_hits > negative_hits:
        label = "positive"
    elif negative_hits > positive_hits:
        label = "negative"
    else:
        label = "neutral"

    debug = {
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "token_count": len(tokens),
    }
    return label, debug


def calculate_entity_f1(
    expected_entities: set[tuple[str, str]],
    actual_entities: set[tuple[str, str]],
) -> tuple[float, float, float]:
    if not expected_entities and not actual_entities:
        return 1.0, 1.0, 1.0
    if not expected_entities or not actual_entities:
        return 0.0, 0.0, 0.0

    overlap = expected_entities.intersection(actual_entities)
    precision = len(overlap) / len(actual_entities)
    recall = len(overlap) / len(expected_entities)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def build_bm25_index(documents: list[str]) -> dict:
    tokenized_docs = [tokenize_words(doc) for doc in documents]
    doc_freq: dict[str, int] = {}
    for tokens in tokenized_docs:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / max(len(tokenized_docs), 1)
    return {
        "tokenized_docs": tokenized_docs,
        "doc_freq": doc_freq,
        "avg_doc_len": avg_doc_len,
        "doc_count": len(tokenized_docs),
    }


def bm25_scores(
    query: str,
    index: dict,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    query_tokens = tokenize_words(query)
    tokenized_docs = index["tokenized_docs"]
    doc_freq = index["doc_freq"]
    avg_doc_len = index["avg_doc_len"] or 1.0
    doc_count = index["doc_count"]

    scores: list[float] = []
    for tokens in tokenized_docs:
        score = 0.0
        doc_len = len(tokens) or 1
        for token in query_tokens:
            tf = tokens.count(token)
            if tf == 0:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf * (numerator / denominator)
        scores.append(round(score, 4))
    return scores


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def fuzzy_ratio(left: str, right: str) -> float:
    return round(SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio(), 4)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def rank_sentences_by_keywords(text: str, priority_terms: set[str], top_k: int = 2) -> list[str]:
    sentences = split_sentences(text)
    scored = []
    for idx, sentence in enumerate(sentences):
        tokens = tokenize_words(sentence)
        keyword_hits = sum(1 for token in tokens if token in priority_terms)
        scored.append((keyword_hits, -idx, sentence))
    ranked = [sentence for _, _, sentence in sorted(scored, reverse=True)[:top_k]]
    return ranked
