import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional


def render_backtest_view(
    feature_df: pd.DataFrame,
    predictions: Optional[np.ndarray] = None,
    label_encoder=None,
    model_metrics: Optional[dict] = None,
):
    st.markdown("## 🔬 Backtest Analysis")

    if feature_df.empty:
        st.warning("No historical data available for backtesting.")
        return

    if model_metrics:
        st.markdown("### Model Performance (Cross-Validated)")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Random Forest**")
            rf = model_metrics.get("rf_metrics", {})
            st.metric("Accuracy", f"{rf.get('accuracy', 0):.1%}")
            st.metric("F1 (Macro)", f"{rf.get('f1_macro', 0):.1%}")

        with col2:
            st.markdown("**Gradient Boosting**")
            gb = model_metrics.get("gb_metrics", {})
            st.metric("Accuracy", f"{gb.get('accuracy', 0):.1%}")
            st.metric("F1 (Macro)", f"{gb.get('f1_macro', 0):.1%}")

        best = model_metrics.get("best_name", "N/A")
        st.success(f"**Best Model:** {best}")

    st.markdown("---")

    st.markdown("### Prediction vs Actual Revisions")

    df = feature_df.copy()
    df = df.sort_values(["symbol", "year", "quarter"])
    df["period"] = df.apply(lambda r: f"Q{r['quarter']} {r['year']}", axis=1)

    if predictions is not None and label_encoder is not None:
        pred_labels = label_encoder.inverse_transform(predictions)
        df["predicted"] = pred_labels[:len(df)]
    elif "predicted" not in df.columns:
        df["predicted"] = "N/A"

    display_df = df[["symbol", "period", "revision_label", "predicted", "revision_pct"]].copy()
    display_df.columns = ["Symbol", "Period", "Actual", "Predicted", "Revision %"]
    display_df["Revision %"] = display_df["Revision %"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    display_df["Correct"] = display_df.apply(
        lambda r: "✅" if r["Actual"] == r["Predicted"] else "❌" if r["Predicted"] != "N/A" else "—",
        axis=1,
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    if "predicted" in df.columns and df["predicted"].iloc[0] != "N/A":
        st.markdown("### Accuracy by Company")
        acc_by_company = df.groupby("symbol", group_keys=False).apply(
            lambda g: pd.Series({"Accuracy": (g["revision_label"] == g["predicted"]).mean()})
        ).reset_index()
        acc_by_company.columns = ["Symbol", "Accuracy"]
        acc_by_company = acc_by_company.sort_values("Accuracy", ascending=True)

        fig = go.Figure(go.Bar(
            x=acc_by_company["Accuracy"],
            y=acc_by_company["Symbol"],
            orientation="h",
            marker_color=acc_by_company["Accuracy"].apply(
                lambda x: "#00C853" if x > 0.6 else "#FF6D00" if x > 0.4 else "#D50000"
            ),
            text=acc_by_company["Accuracy"].apply(lambda x: f"{x:.0%}"),
            textposition="outside",
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=60, t=20, b=20),
            xaxis=dict(range=[0, 1], title="Accuracy"),
        )

        st.plotly_chart(fig, use_container_width=True)

    if "revision_pct" in df.columns and "sentiment_mean" in df.columns:
        st.markdown("### Sentiment vs Actual Revision")

        fig = px.scatter(
            df,
            x="sentiment_mean",
            y="revision_pct",
            color="symbol",
            hover_data=["period"],
            labels={
                "sentiment_mean": "Transcript Sentiment (Mean)",
                "revision_pct": "Actual Consensus Revision %",
            },
        )
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
