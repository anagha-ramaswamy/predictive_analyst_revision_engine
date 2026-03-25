# Predictive Analyst Revision Engine

A prototype that extracts qualitative signals from earnings call transcripts using NLP to estimate whether analyst consensus will revise upward, downward, or hold flat after a call.

See [PROPOSAL.md](PROPOSAL.md) for the full project proposal.

## The Problem

Sell-side consensus estimates are slow to adjust. Analysts anchor on reported numbers and update only after new data forces a change. Earnings calls contain qualitative signals that arrive earlier: hedging language increases before a miss, guidance becomes less specific when management loses confidence, and there is often a gap between the tone of prepared remarks and the tone under questioning.

This system focuses on language, extracting signals that appear before structured updates. To avoid circularity, the model excludes stock prices, headline EPS, and revenue figures. All inputs come from transcript text.

## Features

- **Revision Pressure Score** — Probabilistic score (−1 to +1) estimating consensus revision direction
- **Revision Insight** — Natural-language explanation of what consensus may be missing
- **Transcript Viewer** — Color-coded sentiment and hedging highlighting with section filtering
- **SHAP Explainability** — Waterfall plots decomposing each prediction into feature contributions
- **Sentiment Trajectory** — Sentiment and hedging score tracked across quarters
- **Consensus History** — Before-call vs after-call estimate comparison
- **Backtest Analysis** — Historical prediction accuracy by company

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

The app runs with built-in sample data. No API keys required for the demo.

## Data Integrity

The system never uses the following as model inputs:
- Stock price movements
- Headline EPS or revenue figures
- Analyst target prices or ratings

All features come from the qualitative content of the transcript: tone, hedging, topic emphasis, guidance specificity, and Q&A divergence. Consensus estimates are used only as the prediction target.

## NLP Feature Pipeline

| Feature | Method |
|---------|--------|
| Sentiment trajectory | FinBERT or VADER at the sentence level, aggregated into mean, variance, and QoQ delta |
| Hedging intensity | Frequency of conditional language normalized by sentence count |
| Guidance specificity | Share of forward-looking statements containing concrete numbers |
| Risk term escalation | Changes in use of terms like headwind, pressure, challenging |
| Q&A divergence | Difference between sentiment in prepared remarks and Q&A responses |
| Topic shift | TF-IDF Jaccard distance measuring how management focus changed QoQ |
| Forward-looking ratio | Proportion of sentences classified as forward-looking |

## Model

- Random Forest and Gradient Boosting classifiers with stratified k-fold cross-validation
- Target: revision direction (up / flat / down) based on a 2% consensus change threshold
- SHAP TreeExplainer for per-prediction feature attribution
- Insight generator translating model output into investment reasoning

## Project Structure

```
├── app.py                      Main Streamlit application
├── config.py                   Configuration and constants
├── PROPOSAL.md                 Project proposal
├── requirements.txt            Dependencies
├── data/
│   ├── transcripts.py          FMP API transcript fetcher
│   ├── consensus.py            Consensus estimate fetcher
│   ├── sample_data.py          Built-in demo data
│   ├── analyst_reports.py      Analyst report ingestion (planned)
│   └── sec_filings.py          SEC EDGAR filing ingestion (planned)
├── nlp/
│   ├── preprocessing.py        Transcript splitting and tokenization
│   ├── sentiment.py            FinBERT / VADER sentiment
│   ├── hedging.py              Hedging language detection
│   ├── topics.py               TF-IDF topic analysis
│   └── features.py             Feature matrix builder
├── model/
│   ├── trainer.py              Random Forest and Gradient Boosting
│   ├── scorer.py               Revision Pressure Score
│   ├── explainer.py            SHAP explainability
│   └── insight_generator.py    Natural-language insight
└── ui/
    ├── company_selector.py     Company and quarter selection
    ├── transcript_viewer.py    Highlighted transcript view
    ├── dashboard.py            Score gauge, insight, charts
    └── backtest.py             Historical accuracy analysis
```

## Data Sources

| Source | Status |
|--------|--------|
| Financial Modeling Prep transcripts | Implemented |
| FMP / Alpha Vantage consensus estimates | Implemented |
| Sell-side analyst reports | Planned |
| SEC EDGAR filings (8-K, 10-Q, 10-K) | Planned |

## Tech Stack

Python, Streamlit, Plotly, scikit-learn, SHAP, NLTK, pandas, numpy. FinBERT via HuggingFace Transformers is optional.
