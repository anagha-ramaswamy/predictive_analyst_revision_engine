import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional

from model.scorer import interpret_score, score_color


def render_score_gauge(score: float, probabilities: dict, prediction: str):
    st.markdown("## 📊 Revision Pressure Score")

    color = score_color(score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={"font": {"size": 48}},
        title={"text": "Revision Pressure Score", "font": {"size": 20}},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [-1, -0.3], "color": "rgba(213,0,0,0.1)"},
                {"range": [-0.3, -0.1], "color": "rgba(255,109,0,0.1)"},
                {"range": [-0.1, 0.1], "color": "rgba(255,214,0,0.1)"},
                {"range": [0.1, 0.3], "color": "rgba(100,221,23,0.1)"},
                {"range": [0.3, 1], "color": "rgba(0,200,83,0.1)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    interpretation = interpret_score(score)
    st.markdown(f"**Interpretation:** {interpretation}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("P(Upward)", f"{probabilities.get('up', 0):.1%}")
    with col2:
        st.metric("P(Flat)", f"{probabilities.get('flat', 0):.1%}")
    with col3:
        st.metric("P(Downward)", f"{probabilities.get('down', 0):.1%}")

    st.markdown(f"**Model Prediction:** {prediction.upper()}")


def render_sentiment_trend(feature_history: pd.DataFrame, symbol: str):
    st.markdown("### Sentiment Trajectory")

    df = feature_history[feature_history["symbol"] == symbol].copy()
    if df.empty:
        st.info("No historical data available.")
        return

    df = df.sort_values(["year", "quarter"])
    df["period"] = df.apply(lambda r: f"Q{r['quarter']} {r['year']}", axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["period"],
        y=df["sentiment_mean"],
        mode="lines+markers",
        name="Mean Sentiment",
        line=dict(color="#1976D2", width=2),
        marker=dict(size=8),
    ))

    fig.add_trace(go.Scatter(
        x=df["period"],
        y=df["hedging_score"],
        mode="lines+markers",
        name="Hedging Score",
        line=dict(color="#FF6D00", width=2, dash="dash"),
        marker=dict(size=6),
        yaxis="y2",
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(title="Sentiment", side="left"),
        yaxis2=dict(title="Hedging", side="right", overlaying="y"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_feature_breakdown(features: dict, feature_names: Optional[list[str]] = None):
    st.markdown("### Feature Breakdown")

    display_features = {
        "sentiment_mean": "Avg Sentiment",
        "sentiment_variance": "Sentiment Variance",
        "pct_negative_sentences": "% Negative",
        "sentiment_delta": "Sentiment Δ (vs Prior Q)",
        "hedging_score": "Hedging Score",
        "forward_looking_ratio": "Forward-Looking Ratio",
        "guidance_specificity": "Guidance Specificity",
        "risk_term_frequency": "Risk Term Frequency",
        "topic_shift_score": "Topic Shift",
        "qa_sentiment_gap": "Q&A Sentiment Gap",
    }

    names = []
    values = []
    colors = []

    for key, label in display_features.items():
        val = features.get(key, 0)
        if val is not None:
            names.append(label)
            values.append(val)
            colors.append("#00C853" if val > 0 else "#D50000" if val < 0 else "#9E9E9E")

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=60, t=20, b=20),
        xaxis_title="Value",
        yaxis=dict(autorange="reversed"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_revision_insight(insight: dict):
    st.markdown("### 🔍 Revision Insight")

    st.markdown(f"**{insight['headline']}**")

    if insight["signals"]:
        st.markdown("#### Detected Signals")
        for sig_type, description, direction in insight["signals"]:
            icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}[direction]
            st.markdown(f"{icon} **{sig_type}**: {description}")

    st.markdown("#### Model Reasoning")
    st.markdown(insight["reasoning"])

    with st.expander("⚠️ Risk Factors & Caveats"):
        for rf in insight["risk_factors"]:
            st.markdown(f"- {rf}")

    with st.expander("🔒 Data Integrity & Exclusions"):
        st.markdown(insight["data_exclusions"])


def render_shap_plot(fig):
    st.markdown("### SHAP Feature Contributions")
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("SHAP analysis not available.")


def render_consensus_comparison(
    consensus_df: pd.DataFrame, symbol: str
):
    st.markdown("### Consensus Estimate History")

    df = consensus_df[consensus_df["symbol"] == symbol].copy()
    if df.empty:
        st.info("No consensus data available.")
        return

    df = df.sort_values(["year", "quarter"])
    df["period"] = df.apply(lambda r: f"Q{r['quarter']} {r['year']}", axis=1)

    fig = go.Figure()

    if "estimate_before_call" in df.columns:
        fig.add_trace(go.Bar(
            x=df["period"],
            y=df["estimate_before_call"],
            name="Estimate Before Call",
            marker_color="rgba(25, 118, 210, 0.7)",
        ))

    if "estimate_after_call" in df.columns:
        fig.add_trace(go.Bar(
            x=df["period"],
            y=df["estimate_after_call"],
            name="Estimate After Call",
            marker_color="rgba(0, 200, 83, 0.7)",
        ))

    if "actual_eps" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["period"],
            y=df["actual_eps"],
            name="Actual EPS",
            mode="lines+markers",
            line=dict(color="#FF6D00", width=2),
            marker=dict(size=8),
        ))

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="EPS ($)",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
