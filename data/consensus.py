import requests
import pandas as pd
from typing import Optional
from config import FMP_API_KEY, REVISION_THRESHOLD


FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


def fetch_analyst_estimates(symbol: str) -> Optional[pd.DataFrame]:
    if not FMP_API_KEY:
        return None

    url = f"{FMP_BASE_URL}/analyst-estimates/{symbol}"
    params = {"period": "quarter", "limit": 40, "apikey": FMP_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            df["quarter"] = df["date"].dt.quarter
            df["symbol"] = symbol
            cols = [
                "symbol", "year", "quarter", "date",
                "estimatedEpsAvg", "estimatedEpsHigh", "estimatedEpsLow",
            ]
            available = [c for c in cols if c in df.columns]
            return df[available]
    except Exception:
        pass

    return None


def fetch_earnings_actual(symbol: str) -> Optional[pd.DataFrame]:
    if not FMP_API_KEY:
        return None

    url = f"{FMP_BASE_URL}/earnings-surprises/{symbol}"
    params = {"apikey": FMP_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            df["quarter"] = df["date"].dt.quarter
            df["symbol"] = symbol
            return df
    except Exception:
        pass

    return None


def compute_revision_label(
    estimate_before: float, estimate_after: float
) -> str:
    if estimate_before == 0:
        return "flat"
    pct_change = (estimate_after - estimate_before) / abs(estimate_before)
    if pct_change > REVISION_THRESHOLD:
        return "up"
    elif pct_change < -REVISION_THRESHOLD:
        return "down"
    return "flat"


def compute_revision_magnitude(estimate_before: float, estimate_after: float) -> float:
    if estimate_before == 0:
        return 0.0
    return (estimate_after - estimate_before) / abs(estimate_before)
