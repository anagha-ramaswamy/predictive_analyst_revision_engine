import requests
import pandas as pd
from typing import Optional
from config import FMP_API_KEY


FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


def fetch_transcript(symbol: str, year: int, quarter: int) -> Optional[dict]:
    if not FMP_API_KEY:
        return None

    url = f"{FMP_BASE_URL}/earning_call_transcript/{symbol}"
    params = {"year": year, "quarter": quarter, "apikey": FMP_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data and len(data) > 0:
            return {
                "symbol": symbol,
                "quarter": quarter,
                "year": year,
                "date": data[0].get("date", ""),
                "content": data[0].get("content", ""),
            }
    except Exception:
        pass

    return None


def fetch_all_transcripts(
    symbols: list[str], quarters: list[tuple[int, int]]
) -> pd.DataFrame:
    records = []
    for symbol in symbols:
        for year, quarter in quarters:
            result = fetch_transcript(symbol, year, quarter)
            if result:
                records.append(result)

    if records:
        return pd.DataFrame(records)
    return pd.DataFrame(columns=["symbol", "quarter", "year", "date", "content"])
