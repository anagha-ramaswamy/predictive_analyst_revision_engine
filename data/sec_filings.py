import pandas as pd
from typing import Optional


EDGAR_BASE_URL = "https://efts.sec.gov/LATEST"


def fetch_filing_text(
    symbol: str, filing_type: str = "10-Q", year: int = 2024
) -> Optional[str]:
    return None


def extract_filing_features(text: str, prior_text: Optional[str] = None) -> dict:
    return {}
