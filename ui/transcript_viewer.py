import streamlit as st
import pandas as pd
from typing import Optional


def _sentiment_color(score: float) -> str:
    if score > 0.15:
        return "rgba(0, 200, 83, 0.15)"   # Green
    elif score < -0.15:
        return "rgba(213, 0, 0, 0.15)"    # Red
    else:
        return "rgba(255, 214, 0, 0.08)"  # Light yellow


def _hedging_border(hedging_count: int) -> str:
    if hedging_count > 0:
        return "border-left: 3px solid #FF6D00; padding-left: 8px;"
    return ""


def render_transcript_viewer(
    transcript_content: str,
    sentence_analysis: Optional[list[dict]] = None,
):
    st.markdown("## 📄 Transcript Viewer")

    if not transcript_content:
        st.warning("No transcript available for this selection.")
        return

    if sentence_analysis is None:
        st.markdown(transcript_content)
        return

    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        show_section = st.selectbox(
            "Section", ["All", "Prepared Remarks", "Q&A"],
            key="tv_section"
        )
    with col2:
        highlight_mode = st.selectbox(
            "Highlight", ["Sentiment", "Hedging", "Both"],
            key="tv_highlight"
        )
    with col3:
        signal_filter = st.selectbox(
            "Show", ["All Sentences", "High Signal Only"],
            key="tv_signal"
        )

    st.markdown("---")

    # Legend
    st.markdown(
        '<div style="display:flex;gap:16px;margin-bottom:12px;font-size:0.85em;">'
        '<span style="background:rgba(0,200,83,0.15);padding:2px 8px;border-radius:4px;">Positive</span>'
        '<span style="background:rgba(213,0,0,0.15);padding:2px 8px;border-radius:4px;">Negative</span>'
        '<span style="border-left:3px solid #FF6D00;padding-left:6px;">Hedging</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Filter sentences
    filtered = sentence_analysis
    if show_section == "Prepared Remarks":
        filtered = [s for s in filtered if s.get("section") == "prepared_remarks"]
    elif show_section == "Q&A":
        filtered = [s for s in filtered if s.get("section") == "qa"]

    if signal_filter == "High Signal Only":
        filtered = [
            s for s in filtered
            if abs(s.get("sentiment_score", 0)) > 0.25
            or s.get("hedging_count", 0) > 1
        ]

    if not filtered:
        st.info("No sentences match the current filter.")
        return

    # Render sentences
    html_parts = []
    for sent in filtered:
        text = sent["text"]
        score = sent.get("sentiment_score", 0)
        h_count = sent.get("hedging_count", 0)
        h_keywords = sent.get("hedging_keywords", [])
        temporal = sent.get("temporal", "neutral")
        section = sent.get("section", "")

        # Build style
        styles = []
        if highlight_mode in ("Sentiment", "Both"):
            bg = _sentiment_color(score)
            styles.append(f"background:{bg};")
        if highlight_mode in ("Hedging", "Both"):
            styles.append(_hedging_border(h_count))

        styles.append("padding:4px 8px; margin:2px 0; border-radius:4px; line-height:1.6;")

        # Build tooltip info
        tooltip = f"Sentiment: {score:.3f} | Hedging: {h_count} | {temporal} | {section}"
        if h_keywords:
            tooltip += f" | Keywords: {', '.join(h_keywords)}"

        html = (
            f'<div style="{" ".join(styles)}" title="{tooltip}">'
            f'{text}'
            f'</div>'
        )
        html_parts.append(html)

    st.markdown("\n".join(html_parts), unsafe_allow_html=True)

    # Stats summary
    st.markdown("---")
    st.markdown(f"**Showing {len(filtered)} of {len(sentence_analysis)} sentences**")
