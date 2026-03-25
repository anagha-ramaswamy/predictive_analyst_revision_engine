import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

FMP_API_KEY = os.getenv("FMP_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

COMPANIES = {
    "AAPL": ("Apple Inc.", "Technology"),
    "MSFT": ("Microsoft Corp.", "Technology"),
    "AMZN": ("Amazon.com Inc.", "Consumer Cyclical"),
    "JPM": ("JPMorgan Chase & Co.", "Financial Services"),
    "JNJ": ("Johnson & Johnson", "Healthcare"),
    "XOM": ("Exxon Mobil Corp.", "Energy"),
    "PG": ("Procter & Gamble Co.", "Consumer Defensive"),
    "NVDA": ("NVIDIA Corp.", "Technology"),
    "META": ("Meta Platforms Inc.", "Technology"),
    "UNH": ("UnitedHealth Group Inc.", "Healthcare"),
}

QUARTERS = [
    (2024, 4), (2024, 3), (2024, 2), (2024, 1),
    (2023, 4), (2023, 3), (2023, 2), (2023, 1),
]

HEDGING_KEYWORDS = [
    "may", "could", "might", "uncertain", "uncertainty", "challenging",
    "headwinds", "headwind", "subject to", "risk", "risks", "volatile",
    "volatility", "pressure", "pressures", "cautious", "cautiously",
    "difficult", "difficulties", "potential", "possibly", "perhaps",
]

FORWARD_LOOKING_KEYWORDS = [
    "expect", "expects", "expected", "expecting", "anticipate", "anticipates",
    "anticipated", "going forward", "looking ahead", "outlook", "forecast",
    "project", "projects", "projected", "guidance", "guide", "future",
    "next quarter", "next year", "will", "plan", "plans", "intend", "intends",
]

BACKWARD_LOOKING_KEYWORDS = [
    "last quarter", "previous quarter", "previously", "prior year",
    "year ago", "last year", "prior quarter", "historically", "in the past",
    "was", "were", "had", "achieved", "delivered", "reported",
]

REVISION_THRESHOLD = 0.02

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}
