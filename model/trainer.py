import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from typing import Optional

from config import RANDOM_FOREST_PARAMS, GRADIENT_BOOSTING_PARAMS, COMPANIES


FEATURE_COLUMNS = [
    "sentiment_mean",
    "sentiment_variance",
    "pct_negative_sentences",
    "sentiment_delta",
    "hedging_score",
    "forward_looking_ratio",
    "guidance_specificity",
    "risk_term_frequency",
    "topic_shift_score",
    "qa_sentiment_gap",
]


def prepare_training_data(
    feature_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str]]:
    df = feature_df.dropna(subset=["revision_label"]).copy()
    sector_map = {sym: info[1] for sym, info in COMPANIES.items()}
    df["sector"] = df["symbol"].map(sector_map).fillna("Unknown")
    sector_dummies = pd.get_dummies(df["sector"], prefix="sector", dtype=float)

    feature_cols = FEATURE_COLUMNS.copy()
    X_features = df[feature_cols].fillna(0).values
    X_sectors = sector_dummies.values
    X = np.hstack([X_features, X_sectors])

    all_feature_names = feature_cols + list(sector_dummies.columns)

    le = LabelEncoder()
    y = le.fit_transform(df["revision_label"])

    return X, y, le, all_feature_names


def train_models(
    X: np.ndarray, y: np.ndarray, le: LabelEncoder, feature_names: list[str]
) -> dict:
    unique_classes = np.unique(y)
    class_counts = np.bincount(y)
    min_count = min(class_counts[class_counts > 0])
    n_splits = min(5, min_count)
    n_splits = max(2, n_splits)

    rf = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    try:
        rf_preds = cross_val_predict(rf, X, y, cv=skf)
    except ValueError:
        rf.fit(X, y)
        rf_preds = rf.predict(X)

    rf.fit(X, y)

    rf_metrics = {
        "accuracy": accuracy_score(y, rf_preds),
        "f1_macro": f1_score(y, rf_preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y, rf_preds),
        "report": classification_report(
            y, rf_preds, target_names=le.classes_, zero_division=0, output_dict=True
        ),
    }

    gb = GradientBoostingClassifier(**GRADIENT_BOOSTING_PARAMS)

    try:
        gb_preds = cross_val_predict(gb, X, y, cv=skf)
    except ValueError:
        gb.fit(X, y)
        gb_preds = gb.predict(X)

    gb.fit(X, y)

    gb_metrics = {
        "accuracy": accuracy_score(y, gb_preds),
        "f1_macro": f1_score(y, gb_preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y, gb_preds),
        "report": classification_report(
            y, gb_preds, target_names=le.classes_, zero_division=0, output_dict=True
        ),
    }

    if rf_metrics["f1_macro"] >= gb_metrics["f1_macro"]:
        best_model = rf
        best_name = "Random Forest"
    else:
        best_model = gb
        best_name = "Gradient Boosting"

    return {
        "rf_model": rf,
        "gb_model": gb,
        "rf_metrics": rf_metrics,
        "gb_metrics": gb_metrics,
        "best_model": best_model,
        "best_name": best_name,
        "label_encoder": le,
        "feature_names": feature_names,
    }
