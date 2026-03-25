import re
from config import HEDGING_KEYWORDS


def compute_hedging_score(sentences: list[str]) -> dict:
    if not sentences:
        return {
            "hedging_score": 0.0,
            "hedging_count": 0,
            "hedging_sentences": [],
        }

    total_hedging = 0
    sentences_with_hedging = 0
    flagged_sentences = []

    for sent in sentences:
        lower = sent.lower()
        found_keywords = []
        for kw in HEDGING_KEYWORDS:
            if len(kw) <= 4:
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, lower):
                    found_keywords.append(kw)
            else:
                if kw in lower:
                    found_keywords.append(kw)

        if found_keywords:
            sentences_with_hedging += 1
            total_hedging += len(found_keywords)
            flagged_sentences.append((sent, found_keywords))

    return {
        "hedging_score": sentences_with_hedging / len(sentences),
        "hedging_count": total_hedging,
        "hedging_sentences": flagged_sentences,
    }


def get_hedging_intensity(sentences: list[str]) -> list[dict]:
    results = []
    for sent in sentences:
        lower = sent.lower()
        found = []
        for kw in HEDGING_KEYWORDS:
            if len(kw) <= 4:
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, lower):
                    found.append(kw)
            else:
                if kw in lower:
                    found.append(kw)
        results.append({
            "text": sent,
            "hedging_count": len(found),
            "keywords": found,
        })
    return results
