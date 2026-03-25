import numpy as np
from typing import Optional


def compute_revision_pressure_score(
    model, X: np.ndarray, label_encoder
) -> dict:
    if X.ndim == 1:
        X = X.reshape(1, -1)

    proba = model.predict_proba(X)[0]
    classes = label_encoder.classes_

    prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}

    p_up = prob_dict.get("up", 0.0)
    p_down = prob_dict.get("down", 0.0)
    p_flat = prob_dict.get("flat", 0.0)

    score = p_up - p_down

    prediction = classes[np.argmax(proba)]
    confidence = float(np.max(proba))

    return {
        "score": round(score, 4),
        "probabilities": prob_dict,
        "prediction": prediction,
        "confidence": round(confidence, 4),
    }


def interpret_score(score: float) -> str:
    if score > 0.3:
        return "Strong upward revision pressure — management tone is significantly more positive than consensus implies"
    elif score > 0.1:
        return "Moderate upward revision pressure — some positive signals in management commentary"
    elif score > -0.1:
        return "Neutral — management tone aligns with current consensus expectations"
    elif score > -0.3:
        return "Moderate downward revision pressure — cautious signals detected in management tone"
    else:
        return "Strong downward revision pressure — significant negative divergence from consensus"


def score_color(score: float) -> str:
    if score > 0.2:
        return "#00C853"  # Green
    elif score > 0.05:
        return "#64DD17"  # Light green
    elif score > -0.05:
        return "#FFD600"  # Yellow
    elif score > -0.2:
        return "#FF6D00"  # Orange
    else:
        return "#D50000"  # Red
