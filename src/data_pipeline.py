from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

STATE_CODE_MAP = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}
CODE_TO_STATE_MAP = {v: k for k, v in STATE_CODE_MAP.items()}


@st.cache_data(ttl=900)
def fetch_and_clean_eia_data(api_key):
    """Fetch annual crude production from EIA API and normalize schema."""
    url = f"https://api.eia.gov/v2/petroleum/crd/crpdn/data/?api_key={api_key}&frequency=annual&data[0]=value"
    response = requests.get(url, timeout=20)
    data = response.json()

    if response.status_code != 200 or "response" not in data:
        return pd.DataFrame(), {}

    records = data["response"].get("data", [])
    if not records:
        return pd.DataFrame(), {}

    df = pd.DataFrame(records)
    area_col = "area-name" if "area-name" in df.columns else "areaName"
    clean_df = df[["period", area_col, "value", "product-name", "process-name", "units"]].copy()
    clean_df.columns = ["Year", "Region", "Production_Volume", "Product", "Process", "Units"]
    clean_df["Year"] = clean_df["Year"].astype(int)
    clean_df["Production_Volume"] = pd.to_numeric(clean_df["Production_Volume"], errors="coerce").fillna(0.0)
    clean_df = clean_df[~clean_df["Region"].isin(["U.S.", "United States", "U.S. Total", "USA"])]

    # Keep comparable annual crude field production values only.
    clean_df = clean_df[
        (clean_df["Product"].str.contains("Crude Oil", case=False, na=False))
        & (clean_df["Process"].str.contains("Field Production", case=False, na=False))
        & (clean_df["Units"].isin(["MBBL", "MBBL/D"]))
    ].copy()

    # Handle both full-state names and EIA-coded names like USA-TX.
    clean_df["State_Code"] = clean_df["Region"].str.extract(r"USA-([A-Z]{2})", expand=False)
    clean_df["State_Code"] = clean_df["State_Code"].fillna(clean_df["Region"].map(STATE_CODE_MAP))
    clean_df = clean_df.dropna(subset=["State_Code"]).copy()
    clean_df["Region"] = clean_df["State_Code"].map(CODE_TO_STATE_MAP).fillna(clean_df["Region"])

    clean_df = clean_df[["Year", "Region", "State_Code", "Production_Volume"]]
    clean_df = clean_df.groupby(["Year", "Region", "State_Code"], as_index=False)["Production_Volume"].sum()

    metadata = {
        "source": "EIA Open Data API v2 - petroleum/crd/crpdn",
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Annual crude oil production by state.",
    }
    return clean_df, metadata