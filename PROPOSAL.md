# Predictive Analyst Revision Engine

## The Problem

Sell-side consensus estimates are slow to adjust. Analysts tend to anchor on reported numbers and update only after new data forces a change. Yet earnings calls consistently contain qualitative signals that come earlier. Hedging language increases before a miss, guidance becomes less specific when management loses confidence, and there is often a gap between the tone of prepared remarks and the tone under questioning.

At key inflection points, this creates a window where consensus does not reflect information already present in management's own words. Prior research, including Loughran and McDonald (2011), shows that linguistic features in disclosures can predict earnings outcomes. Still, most systematic pipelines underuse transcript-level information.

Traditional quant signals rely on structured inputs such as earnings, price movement, or revisions themselves. This system instead focuses on language, extracting signals that appear before those structured updates.

## The System

I propose a Revision Pressure Score that estimates whether consensus forecasts will move up, down, or remain unchanged after an earnings call. To avoid circularity, the model excludes stock prices, headline EPS, and revenue figures. All inputs come from transcript text.

The pipeline extracts several types of features:

Sentiment trajectory using FinBERT at the sentence level, aggregated into summary statistics and compared across quarters, with separate treatment of prepared remarks and Q and A.

Hedging intensity measured as the frequency of conditional language normalized by sentence count.

Guidance specificity captured as the share of forward-looking statements that include concrete numbers rather than vague language.

Risk term escalation based on changes in the use of words such as headwind, pressure, and challenging.

Q and A divergence defined as the difference between sentiment in prepared remarks and responses under questioning.

Topic shift using TF-IDF based similarity to measure how management's focus has changed relative to the prior quarter.

These features feed tree-based models such as Random Forest and Gradient Boosting trained on historical transcript and revision pairs. SHAP values provide interpretability, allowing each prediction to be traced back to specific features and language patterns.

## Alpha Generation

The signal becomes useful when it disagrees with stable consensus expectations. A strong negative revision signal without corresponding analyst movement suggests a potential short. A positive signal highlights situations where improvements are not yet reflected in forecasts.

The model can be applied across a universe of stocks to rank names by revision pressure and construct a long short portfolio with limited exposure to sector or market beta. Returns are realized as analyst revisions and prices adjust to reflect information that was already present in the transcripts.

## The Insight

Markets incorporate reported numbers quickly but are slower to react to shifts in tone and confidence. This system treats changes in narrative as measurable signals and captures early stages of expectation changes before they appear in forecasts or prices. The goal is not only prediction but also understanding which changes in language reflect real business shifts rather than temporary noise.

The system can be extended to include analyst reports and SEC filings such as 8-K and 10-Q documents to broaden the information set.

## Technical Requirements

Data can be sourced from Financial Modeling Prep or SEC EDGAR for transcripts, along with consensus estimate data from providers such as FMP or Alpha Vantage.

The implementation uses Python with libraries including scikit-learn, shap, nltk, pandas, plotly, and streamlit. Transformer models such as FinBERT can be added using PyTorch.

## Anticipated Risks

A small initial dataset may lead to overfitting. A prototype could be built on a limited set of firms, but a robust system would require a larger cross section and multiple market environments.

Sentiment models may miss nuance such as sarcasm or strategic phrasing, especially in Q and A sections.

Language patterns differ across sectors and macro conditions, which may reduce stability across regimes.

Transcript availability from free sources may introduce delays, limiting real-time use without higher quality data feeds.
