import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional


RISK_TERMS = [
    "risk", "risks", "headwind", "headwinds", "pressure", "pressures",
    "challenge", "challenges", "challenging", "uncertain", "uncertainty",
    "volatile", "volatility", "downturn", "recession", "decline",
    "weakness", "weak", "deterioration", "deteriorate", "loss", "losses",
]


def extract_topics(
    texts: list[str], n_top: int = 15, max_features: int = 500
) -> dict:
    if not texts:
        return {
            "top_terms": [],
            "tfidf_matrix": None,
            "feature_names": [],
            "vectorizer": None,
        }

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = avg_scores.argsort()[-n_top:][::-1]
    top_terms = [(feature_names[i], float(avg_scores[i])) for i in top_indices]

    return {
        "top_terms": top_terms,
        "tfidf_matrix": tfidf_matrix,
        "feature_names": feature_names,
        "vectorizer": vectorizer,
    }


def compute_risk_term_frequency(sentences: list[str]) -> dict:
    if not sentences:
        return {
            "risk_term_frequency": 0.0,
            "risk_term_count": 0,
            "risk_sentences": [],
        }

    total_risk = 0
    sentences_with_risk = 0
    flagged = []

    for sent in sentences:
        lower = sent.lower()
        found = [t for t in RISK_TERMS if t in lower]
        if found:
            sentences_with_risk += 1
            total_risk += len(found)
            flagged.append((sent, found))

    return {
        "risk_term_frequency": sentences_with_risk / len(sentences),
        "risk_term_count": total_risk,
        "risk_sentences": flagged,
    }


def compute_topic_shift(
    current_terms: list[tuple[str, float]],
    prior_terms: list[tuple[str, float]],
) -> float:
    if not current_terms or not prior_terms:
        return 0.0

    current_set = set(t[0] for t in current_terms)
    prior_set = set(t[0] for t in prior_terms)

    if not current_set and not prior_set:
        return 0.0

    intersection = current_set & prior_set
    union = current_set | prior_set

    jaccard_distance = 1.0 - len(intersection) / len(union) if union else 0.0

    return round(jaccard_distance, 4)
