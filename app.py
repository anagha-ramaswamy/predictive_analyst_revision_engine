import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

from config import COMPANIES
from data.sample_data import (
    get_sample_transcript,
    get_all_sample_transcripts,
    get_sample_consensus,
    get_sample_features,
    get_available_companies,
)
from data.transcripts import fetch_transcript
from ui.company_selector import render_company_selector
from ui.transcript_viewer import render_transcript_viewer
from ui.dashboard import (
    render_score_gauge,
    render_sentiment_trend,
    render_feature_breakdown,
    render_shap_plot,
    render_consensus_comparison,
    render_revision_insight,
)
from ui.backtest import render_backtest_view


st.set_page_config(
    page_title="Earnings Sentiment Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.15rem !important; }
    .stMetric { background: rgba(28,131,225,0.05); border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "model_trained": False,
        "model_results": None,
        "feature_df": None,
        "consensus_df": None,
        "use_live_api": False,
        "force_vader": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


@st.cache_resource(show_spinner="Loading data and training model...")
def load_and_train(force_vader: bool = True):
    from model.trainer import prepare_training_data, train_models, FEATURE_COLUMNS

    feature_df = get_sample_features()
    consensus_df = get_sample_consensus()

    merged = feature_df.merge(
        consensus_df[["symbol", "year", "quarter", "revision_pct",
                       "estimate_before_call", "estimate_after_call", "actual_eps"]],
        on=["symbol", "year", "quarter"],
        how="left",
    )

    def label_revision(pct):
        if pd.isna(pct):
            return "flat"
        if pct > 0.02:
            return "up"
        elif pct < -0.02:
            return "down"
        return "flat"

    merged["revision_label"] = merged["revision_pct"].apply(label_revision)

    X, y, le, feature_names = prepare_training_data(merged)
    model_results = train_models(X, y, le, feature_names)

    predictions = model_results["best_model"].predict(X)
    merged["predicted"] = le.inverse_transform(predictions)

    return merged, consensus_df, model_results, X, y, predictions


def run_live_nlp(content: str, force_vader: bool = True) -> dict:
    from nlp.preprocessing import process_transcript
    from nlp.sentiment import analyze_sentiment
    from nlp.hedging import get_hedging_intensity

    processed = process_transcript(content)
    sentences = processed["sentences"]
    texts = [s["text"] for s in sentences]

    sentiment = analyze_sentiment(texts, force_vader=force_vader)
    hedging = get_hedging_intensity(texts)

    analysis = []
    for i, sent_info in enumerate(sentences):
        analysis.append({
            "text": sent_info["text"],
            "section": sent_info["section"],
            "temporal": sent_info["temporal"],
            "sentiment_score": sentiment[i]["score"] if i < len(sentiment) else 0,
            "positive": sentiment[i].get("positive", 0),
            "negative": sentiment[i].get("negative", 0),
            "hedging_count": hedging[i]["hedging_count"] if i < len(hedging) else 0,
            "hedging_keywords": hedging[i]["keywords"] if i < len(hedging) else [],
        })

    return {
        "sentences": analysis,
        "processed": processed,
    }


st.sidebar.markdown("# 📈 Earnings Sentiment Tracker")
st.sidebar.markdown("*Revision Pressure from NLP*")
st.sidebar.markdown("---")

symbol, year, quarter = render_company_selector()

st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ Settings")
force_vader = st.sidebar.checkbox(
    "Use VADER (fast, no GPU)",
    value=True,
    help="Use VADER sentiment instead of FinBERT. Faster but less accurate for financial text.",
)
st.session_state["force_vader"] = force_vader

run_live_nlp_toggle = st.sidebar.checkbox(
    "Run live NLP on transcript",
    value=True,
    help="Run sentiment & hedging analysis on the selected transcript in real-time.",
)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Transcript Viewer", "Backtest Analysis"],
    index=0,
)

feature_df, consensus_df, model_results, X_all, y_all, all_predictions = load_and_train(
    force_vader=force_vader,
)
st.session_state["feature_df"] = feature_df
st.session_state["model_results"] = model_results
st.session_state["consensus_df"] = consensus_df

current_features = feature_df[
    (feature_df["symbol"] == symbol)
    & (feature_df["year"] == year)
    & (feature_df["quarter"] == quarter)
]

transcript_data = get_sample_transcript(symbol, year, quarter)
transcript_content = transcript_data["content"] if transcript_data else None

current_score_info = None
if not current_features.empty:
    from model.scorer import compute_revision_pressure_score
    from model.trainer import FEATURE_COLUMNS

    row = current_features.iloc[0]
    sector_map = {sym: info[1] for sym, info in COMPANIES.items()}
    sector = sector_map.get(symbol, "Unknown")

    feature_vals = [row.get(c, 0) for c in FEATURE_COLUMNS]
    all_sectors = sorted(set(sector_map.values()))
    sector_dummies = [1.0 if s == sector else 0.0 for s in all_sectors]
    x_current = np.array(feature_vals + sector_dummies).reshape(1, -1)

    current_score_info = compute_revision_pressure_score(
        model_results["best_model"],
        x_current,
        model_results["label_encoder"],
    )

sentence_analysis = None
if run_live_nlp_toggle and transcript_content:
    nlp_result = run_live_nlp(transcript_content, force_vader=force_vader)
    sentence_analysis = nlp_result["sentences"]


if page == "Dashboard":
    st.markdown(f"# 📈 {COMPANIES.get(symbol, (symbol,))[0]} — Q{quarter} {year}")

    if current_score_info:
        render_score_gauge(
            current_score_info["score"],
            current_score_info["probabilities"],
            current_score_info["prediction"],
        )
    else:
        st.warning("No score available for this selection.")

    if current_score_info and not current_features.empty:
        st.markdown("---")
        from model.insight_generator import generate_revision_insight
        insight = generate_revision_insight(
            features=current_features.iloc[0].to_dict(),
            score=current_score_info["score"],
            prediction=current_score_info["prediction"],
            probabilities=current_score_info["probabilities"],
            symbol=symbol,
            quarter=quarter,
            year=year,
        )
        render_revision_insight(insight)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        render_sentiment_trend(feature_df, symbol)

    with col2:
        if not current_features.empty:
            render_feature_breakdown(current_features.iloc[0].to_dict())

    st.markdown("---")

    render_consensus_comparison(consensus_df, symbol)

    st.markdown("---")
    if not current_features.empty and model_results:
        try:
            from model.explainer import compute_shap_values, create_waterfall_figure

            shap_result = compute_shap_values(
                model_results["best_model"],
                X_all,
                model_results["feature_names"],
            )

            idx = current_features.index[0]
            df_indices = feature_df.index.tolist()
            if idx in df_indices:
                pos = df_indices.index(idx)
                classes = list(model_results["label_encoder"].classes_)
                class_idx = classes.index("up") if "up" in classes else 0

                fig = create_waterfall_figure(
                    shap_result["shap_values"],
                    shap_result["expected_value"],
                    model_results["feature_names"],
                    sample_idx=pos,
                    class_idx=class_idx,
                )
                render_shap_plot(fig)
        except Exception as e:
            st.info(f"SHAP analysis unavailable: {e}")

elif page == "Transcript Viewer":
    st.markdown(f"# 📄 {COMPANIES.get(symbol, (symbol,))[0]} — Q{quarter} {year}")

    if transcript_content:
        render_transcript_viewer(transcript_content, sentence_analysis)
    else:
        st.warning(
            f"No transcript available for {symbol} Q{quarter} {year}. "
            f"Transcripts are available for: AAPL, MSFT, JPM, NVDA, XOM."
        )

elif page == "Backtest Analysis":
    render_backtest_view(
        feature_df,
        all_predictions,
        model_results["label_encoder"],
        model_results,
    )


st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Predictive Analyst Revision Engine</small>",
    unsafe_allow_html=True,
)
