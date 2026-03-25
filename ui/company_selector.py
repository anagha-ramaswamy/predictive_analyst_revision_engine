import streamlit as st
from config import COMPANIES
from data.sample_data import get_available_companies, get_available_quarters


def render_company_selector() -> tuple:
    st.sidebar.markdown("## Select Company & Quarter")

    available = get_available_companies()
    company_options = {}
    for sym in available:
        if sym in COMPANIES:
            name, sector = COMPANIES[sym]
            company_options[f"{sym} — {name} ({sector})"] = sym
        else:
            company_options[sym] = sym

    selected_label = st.sidebar.selectbox(
        "Company",
        options=list(company_options.keys()),
        index=0,
    )
    symbol = company_options[selected_label]

    quarters = get_available_quarters(symbol)
    quarter_labels = {f"Q{q} {y}": (y, q) for y, q in quarters}

    selected_quarter = st.sidebar.selectbox(
        "Quarter",
        options=list(quarter_labels.keys()),
        index=0,
    )
    year, quarter = quarter_labels[selected_quarter]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Selected:** {symbol} — Q{quarter} {year}"
    )

    return symbol, year, quarter
