import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional
import io


def compute_shap_values(
    model, X: np.ndarray, feature_names: list[str], X_background: Optional[np.ndarray] = None
) -> dict:
    if X_background is None:
        X_background = X

    explainer = shap.TreeExplainer(model, X_background)
    shap_values = explainer.shap_values(X)

    return {
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "feature_names": feature_names,
    }


def create_waterfall_figure(
    shap_values, expected_value, feature_names: list[str],
    sample_idx: int = 0, class_idx: int = 0, max_display: int = 12
) -> plt.Figure:
    if isinstance(shap_values, list):
        sv = shap_values[class_idx][sample_idx]
        base = expected_value[class_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    elif shap_values.ndim == 3:
        sv = shap_values[sample_idx, :, class_idx]
        base = expected_value[class_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    else:
        sv = shap_values[sample_idx]
        base = expected_value if not isinstance(expected_value, (list, np.ndarray)) else expected_value[0]

    explanation = shap.Explanation(
        values=sv,
        base_values=base,
        feature_names=feature_names,
    )

    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.tight_layout()
    return fig


def create_summary_figure(
    shap_values, feature_names: list[str], X: np.ndarray,
    class_idx: int = 0, max_display: int = 12
) -> plt.Figure:
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, class_idx]
    else:
        sv = shap_values

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.summary_plot(
        sv, X, feature_names=feature_names,
        max_display=max_display, show=False
    )
    plt.tight_layout()
    return fig


def get_feature_importance_from_shap(
    shap_values, feature_names: list[str], class_idx: int = 0
) -> list[tuple[str, float]]:
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, class_idx]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    pairs = list(zip(feature_names, mean_abs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs
