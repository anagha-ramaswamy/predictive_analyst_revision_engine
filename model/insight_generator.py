from typing import Optional


THRESHOLDS = {
    "sentiment_high": 0.15,
    "sentiment_low": -0.05,
    "hedging_high": 0.18,
    "hedging_low": 0.08,
    "guidance_specific_high": 0.45,
    "guidance_specific_low": 0.25,
    "risk_freq_high": 0.12,
    "risk_freq_low": 0.05,
    "qa_gap_positive": 0.05,
    "qa_gap_negative": -0.05,
    "sentiment_delta_up": 0.06,
    "sentiment_delta_down": -0.06,
    "forward_ratio_high": 0.40,
    "forward_ratio_low": 0.20,
    "topic_shift_high": 0.25,
    "pct_negative_high": 0.30,
}


def generate_revision_insight(
    features: dict,
    score: float,
    prediction: str,
    probabilities: dict,
    symbol: str = "",
    quarter: int = 0,
    year: int = 0,
) -> dict:
    signals = []
    reasoning_parts = []

    sent_mean = features.get("sentiment_mean", 0)
    sent_var = features.get("sentiment_variance", 0)
    sent_delta = features.get("sentiment_delta", 0)
    pct_neg = features.get("pct_negative_sentences", 0)
    hedge = features.get("hedging_score", 0)
    fwd_ratio = features.get("forward_looking_ratio", 0)
    guidance = features.get("guidance_specificity", 0)
    risk_freq = features.get("risk_term_frequency", 0)
    topic_shift = features.get("topic_shift_score", 0)
    qa_gap = features.get("qa_sentiment_gap", 0)

    if sent_mean > THRESHOLDS["sentiment_high"]:
        signals.append((
            "POSITIVE_TONE",
            f"Management tone is notably positive (sentiment: {sent_mean:.3f}), "
            f"suggesting confidence beyond what headline numbers convey.",
            "bullish",
        ))
    elif sent_mean < THRESHOLDS["sentiment_low"]:
        signals.append((
            "NEGATIVE_TONE",
            f"Management tone is unusually cautious or negative (sentiment: {sent_mean:.3f}), "
            f"which may signal headwinds not yet reflected in consensus.",
            "bearish",
        ))

    if sent_delta > THRESHOLDS["sentiment_delta_up"]:
        signals.append((
            "TONE_IMPROVEMENT",
            f"Sentiment improved {sent_delta:+.3f} vs. prior quarter — "
            f"management is more optimistic than last call, possibly signaling inflection.",
            "bullish",
        ))
    elif sent_delta < THRESHOLDS["sentiment_delta_down"]:
        signals.append((
            "TONE_DETERIORATION",
            f"Sentiment declined {sent_delta:+.3f} vs. prior quarter — "
            f"management outlook has worsened, a leading indicator of estimate cuts.",
            "bearish",
        ))

    if hedge > THRESHOLDS["hedging_high"]:
        signals.append((
            "HIGH_HEDGING",
            f"Hedging language is elevated ({hedge:.1%} of sentences), with frequent use of "
            f"conditional terms ('may', 'could', 'uncertain'). Management is embedding "
            f"optionality into forward statements — a classic pre-revision pattern.",
            "bearish",
        ))
    elif hedge < THRESHOLDS["hedging_low"]:
        signals.append((
            "LOW_HEDGING",
            f"Hedging language is unusually low ({hedge:.1%} of sentences). "
            f"Management is making direct, unconditional forward statements — "
            f"consistent with high conviction in near-term outlook.",
            "bullish",
        ))

    if guidance > THRESHOLDS["guidance_specific_high"]:
        signals.append((
            "SPECIFIC_GUIDANCE",
            f"Guidance specificity is high ({guidance:.1%} of forward-looking sentences "
            f"contain concrete numbers). Specific quantitative guidance correlates with "
            f"management confidence and lower post-call revision volatility.",
            "bullish",
        ))
    elif guidance < THRESHOLDS["guidance_specific_low"]:
        signals.append((
            "VAGUE_GUIDANCE",
            f"Guidance specificity is low ({guidance:.1%}). Management is relying on "
            f"qualitative language rather than concrete figures — often a precursor to "
            f"wider-than-expected outcome ranges.",
            "bearish",
        ))

    if risk_freq > THRESHOLDS["risk_freq_high"]:
        signals.append((
            "ELEVATED_RISK_LANGUAGE",
            f"Risk-related language appears in {risk_freq:.1%} of sentences — "
            f"significantly above baseline. Terms like 'headwind', 'pressure', and "
            f"'challenging' are surfacing at elevated rates.",
            "bearish",
        ))

    if qa_gap < THRESHOLDS["qa_gap_negative"]:
        signals.append((
            "QA_SENTIMENT_DROP",
            f"Sentiment drops {abs(qa_gap):.3f} between prepared remarks and Q&A. "
            f"When analysts probe, management's tone becomes more cautious — "
            f"prepared remarks may be overly optimistic relative to true outlook.",
            "bearish",
        ))
    elif qa_gap > THRESHOLDS["qa_gap_positive"]:
        signals.append((
            "QA_SENTIMENT_LIFT",
            f"Sentiment improves {qa_gap:.3f} during Q&A vs. prepared remarks. "
            f"Management becomes more positive under analyst questioning — "
            f"suggesting the scripted remarks understated management confidence.",
            "bullish",
        ))

    if topic_shift > THRESHOLDS["topic_shift_high"]:
        signals.append((
            "TOPIC_SHIFT",
            f"Topic distribution shifted significantly ({topic_shift:.1%} Jaccard distance "
            f"vs. prior quarter). A major change in the topics management emphasizes "
            f"can signal a strategic pivot not yet priced into consensus models.",
            "neutral",
        ))

    if fwd_ratio > THRESHOLDS["forward_ratio_high"]:
        signals.append((
            "FORWARD_FOCUS",
            f"An unusually high proportion of sentences ({fwd_ratio:.1%}) are forward-looking. "
            f"Management is steering the narrative toward the future, which often "
            f"precedes a guidance inflection.",
            "neutral",
        ))
    elif fwd_ratio < THRESHOLDS["forward_ratio_low"]:
        signals.append((
            "BACKWARD_FOCUS",
            f"Only {fwd_ratio:.1%} of sentences are forward-looking — management "
            f"spent more time discussing past results than guiding for the future. "
            f"This defensive posture can signal reluctance to commit to estimates.",
            "bearish",
        ))

    bullish_count = sum(1 for _, _, d in signals if d == "bullish")
    bearish_count = sum(1 for _, _, d in signals if d == "bearish")

    if score > 0.2 and bullish_count > bearish_count:
        headline = (
            f"Upward revision pressure detected for {symbol} Q{quarter} {year}: "
            f"management tone diverges positively from consensus."
        )
    elif score < -0.2 and bearish_count > bullish_count:
        headline = (
            f"Downward revision pressure detected for {symbol} Q{quarter} {year}: "
            f"cautious signals suggest consensus is too optimistic."
        )
    elif abs(score) < 0.1:
        headline = (
            f"No significant revision pressure for {symbol} Q{quarter} {year}: "
            f"management tone aligns with consensus expectations."
        )
    else:
        headline = (
            f"Mixed signals for {symbol} Q{quarter} {year}: "
            f"model detects {bullish_count} bullish and {bearish_count} bearish indicators."
        )

    if signals:
        reasoning_parts.append(
            f"Analysis of the {symbol} Q{quarter} {year} earnings call transcript "
            f"identified {len(signals)} notable signals that may not be fully reflected "
            f"in current consensus estimates:"
        )
        for sig_type, description, direction in signals:
            direction_label = {"bullish": "↑", "bearish": "↓", "neutral": "→"}[direction]
            reasoning_parts.append(f"  {direction_label} **{sig_type}**: {description}")

        reasoning_parts.append("")
        reasoning_parts.append(
            f"The model assigns a Revision Pressure Score of {score:+.3f} "
            f"(P(up)={probabilities.get('up', 0):.1%}, "
            f"P(flat)={probabilities.get('flat', 0):.1%}, "
            f"P(down)={probabilities.get('down', 0):.1%}), "
            f"predicting a **{prediction}** revision to consensus estimates."
        )
    else:
        reasoning_parts.append(
            f"No strong divergence signals detected in the {symbol} Q{quarter} {year} "
            f"earnings call. Feature values fall within normal ranges, suggesting "
            f"current consensus is well-calibrated to management's communicated outlook."
        )

    risk_factors = [
        "This analysis is based solely on qualitative features extracted from the "
        "earnings call transcript. Exogenous shocks, sector rotation, and macro "
        "developments may override transcript-derived signals.",
        "The model is trained on historical revision patterns which may not persist "
        "in regime changes (e.g., rate cycle shifts, sector re-ratings).",
        "Sentiment analysis may not capture sarcasm, deflection, or culturally "
        "specific communication styles.",
    ]

    if sent_var > 0.08:
        risk_factors.append(
            f"High sentiment variance ({sent_var:.3f}) suggests mixed signals within "
            f"the transcript. The aggregate score may mask important within-call divergences."
        )

    data_exclusions = (
        "**Data Integrity Note:** This analysis deliberately excludes stock price "
        "movements, headline EPS figures, and revenue beats/misses to avoid circular "
        "logic. All features are derived exclusively from the qualitative content "
        "of the earnings call transcript — specifically management tone, language "
        "patterns, hedging behavior, topic emphasis, and guidance specificity. "
        "This ensures the system identifies information that consensus models, "
        "which typically anchor on reported numbers, may be slow to incorporate."
    )

    return {
        "headline": headline,
        "signals": signals,
        "reasoning": "\n\n".join(reasoning_parts),
        "risk_factors": risk_factors,
        "data_exclusions": data_exclusions,
    }
