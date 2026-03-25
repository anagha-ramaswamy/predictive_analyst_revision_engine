
import numpy as np
from typing import Optional


_finbert_pipeline = None
_vader_analyzer = None
_use_finbert = None


def _load_finbert():
    global _finbert_pipeline, _use_finbert
    try:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        _use_finbert = True
    except Exception:
        _use_finbert = False


def _load_vader():
    global _vader_analyzer
    try:
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    except Exception:
        pass


def _get_analyzer():
    global _use_finbert
    if _use_finbert is None:
        _load_finbert()
    if not _use_finbert and _vader_analyzer is None:
        _load_vader()
    return _use_finbert


def analyze_sentiment_finbert(sentences: list[str]) -> list[dict]:
    global _finbert_pipeline
    if _finbert_pipeline is None:
        _load_finbert()

    results = []
    batch_size = 32
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch = [s[:512] for s in batch]
        try:
            outputs = _finbert_pipeline(batch)
            for sent, output in zip(batch, outputs):
                scores = {item["label"]: item["score"] for item in output}
                results.append({
                    "text": sent,
                    "positive": scores.get("positive", 0),
                    "negative": scores.get("negative", 0),
                    "neutral": scores.get("neutral", 0),
                    "score": scores.get("positive", 0) - scores.get("negative", 0),
                })
        except Exception:
            for sent in batch:
                results.append({
                    "text": sent,
                    "positive": 0.33,
                    "negative": 0.33,
                    "neutral": 0.34,
                    "score": 0.0,
                })
    return results


def analyze_sentiment_vader(sentences: list[str]) -> list[dict]:
    global _vader_analyzer
    if _vader_analyzer is None:
        _load_vader()

    results = []
    for sent in sentences:
        if _vader_analyzer:
            scores = _vader_analyzer.polarity_scores(sent)
            results.append({
                "text": sent,
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
                "score": scores["compound"],  # -1 to 1
            })
        else:
            results.append({
                "text": sent,
                "positive": 0.33,
                "negative": 0.33,
                "neutral": 0.34,
                "score": 0.0,
            })
    return results


def analyze_sentiment(sentences: list[str], force_vader: bool = False) -> list[dict]:
    if force_vader:
        return analyze_sentiment_vader(sentences)

    use_finbert = _get_analyzer()
    if use_finbert:
        return analyze_sentiment_finbert(sentences)
    else:
        return analyze_sentiment_vader(sentences)


def compute_sentiment_features(sentiment_results: list[dict]) -> dict:
    if not sentiment_results:
        return {
            "sentiment_mean": 0.0,
            "sentiment_variance": 0.0,
            "pct_negative_sentences": 0.0,
            "pct_positive_sentences": 0.0,
            "sentiment_range": 0.0,
        }

    scores = [r["score"] for r in sentiment_results]
    arr = np.array(scores)

    return {
        "sentiment_mean": float(np.mean(arr)),
        "sentiment_variance": float(np.var(arr)),
        "pct_negative_sentences": float(np.mean(arr < -0.1)),
        "pct_positive_sentences": float(np.mean(arr > 0.1)),
        "sentiment_range": float(np.max(arr) - np.min(arr)),
    }
