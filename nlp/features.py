import re
import numpy as np
import pandas as pd
from typing import Optional

from nlp.preprocessing import process_transcript
from nlp.sentiment import analyze_sentiment, compute_sentiment_features
from nlp.hedging import compute_hedging_score
from nlp.topics import extract_topics, compute_risk_term_frequency, compute_topic_shift


def compute_guidance_specificity(sentences: list[str]) -> float:
    if not sentences:
        return 0.0

    specific_count = 0
    number_pattern = re.compile(
        r"\$[\d,]+\.?\d*|\d+\.?\d*\s*%|\d+\.?\d*\s*(?:billion|million|thousand|bps|basis points)"
        r"|\d{1,3}(?:,\d{3})+",
        re.IGNORECASE,
    )

    for sent in sentences:
        if number_pattern.search(sent):
            specific_count += 1

    return specific_count / len(sentences)


def extract_features_from_transcript(
    content: str,
    force_vader: bool = False,
    prior_topics: Optional[list[tuple[str, float]]] = None,
) -> dict:
    processed = process_transcript(content)
    sentences = processed["sentences"]
    sentence_texts = [s["text"] for s in sentences]

    prepared_texts = [s["text"] for s in sentences if s["section"] == "prepared_remarks"]
    qa_texts = [s["text"] for s in sentences if s["section"] == "qa"]
    forward_texts = [s["text"] for s in sentences if s["temporal"] == "forward"]

    all_sentiment = analyze_sentiment(sentence_texts, force_vader=force_vader)
    sentiment_features = compute_sentiment_features(all_sentiment)

    if prepared_texts:
        prepared_sentiment = analyze_sentiment(prepared_texts, force_vader=force_vader)
        prepared_features = compute_sentiment_features(prepared_sentiment)
    else:
        prepared_features = {"sentiment_mean": 0.0}

    if qa_texts:
        qa_sentiment = analyze_sentiment(qa_texts, force_vader=force_vader)
        qa_features = compute_sentiment_features(qa_sentiment)
    else:
        qa_features = {"sentiment_mean": 0.0}

    qa_sentiment_gap = qa_features["sentiment_mean"] - prepared_features["sentiment_mean"]

    hedging = compute_hedging_score(sentence_texts)

    topics = extract_topics(prepared_texts if prepared_texts else sentence_texts)
    risk = compute_risk_term_frequency(sentence_texts)

    topic_shift = compute_topic_shift(topics["top_terms"], prior_topics) if prior_topics else 0.0

    guidance_spec = compute_guidance_specificity(forward_texts if forward_texts else sentence_texts)

    forward_ratio = len(forward_texts) / len(sentence_texts) if sentence_texts else 0.0

    features = {
        "sentiment_mean": sentiment_features["sentiment_mean"],
        "sentiment_variance": sentiment_features["sentiment_variance"],
        "pct_negative_sentences": sentiment_features["pct_negative_sentences"],
        "sentiment_delta": 0.0,
        "hedging_score": hedging["hedging_score"],
        "forward_looking_ratio": forward_ratio,
        "guidance_specificity": guidance_spec,
        "risk_term_frequency": risk["risk_term_frequency"],
        "topic_shift_score": topic_shift,
        "num_sentences": len(sentence_texts),
        "qa_sentiment_gap": qa_sentiment_gap,
    }

    return features


def build_feature_matrix(
    transcripts_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    force_vader: bool = False,
) -> pd.DataFrame:
    records = []

    transcripts_df = transcripts_df.sort_values(["symbol", "year", "quarter"])
    prior_topics = {}

    for _, row in transcripts_df.iterrows():
        symbol = row["symbol"]
        year = row["year"]
        quarter = row["quarter"]
        content = row["content"]

        prior = prior_topics.get(symbol)
        features = extract_features_from_transcript(
            content, force_vader=force_vader, prior_topics=prior
        )

        from nlp.preprocessing import process_transcript as _pt
        processed = _pt(content)
        prepared_texts = [
            s["text"] for s in processed["sentences"]
            if s["section"] == "prepared_remarks"
        ]
        topics = extract_topics(prepared_texts if prepared_texts else [s["text"] for s in processed["sentences"]])
        prior_topics[symbol] = topics["top_terms"]

        features["symbol"] = symbol
        features["year"] = year
        features["quarter"] = quarter

        match = consensus_df[
            (consensus_df["symbol"] == symbol)
            & (consensus_df["year"] == year)
            & (consensus_df["quarter"] == quarter)
        ]
        if not match.empty:
            rev_pct = match.iloc[0].get("revision_pct", 0.0)
            features["revision_pct"] = rev_pct
            if rev_pct > 0.02:
                features["revision_label"] = "up"
            elif rev_pct < -0.02:
                features["revision_label"] = "down"
            else:
                features["revision_label"] = "flat"
        else:
            features["revision_pct"] = 0.0
            features["revision_label"] = "flat"

        records.append(features)

    df = pd.DataFrame(records)

    df = df.sort_values(["symbol", "year", "quarter"])
    df["sentiment_delta"] = df.groupby("symbol")["sentiment_mean"].diff().fillna(0)

    return df
