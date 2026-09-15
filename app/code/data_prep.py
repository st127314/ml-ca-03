"""A1/A2 cleaning reused for A3, plus leakage-safe price buckets."""

import numpy as np
import pandas as pd

KM_DRIVEN_LIMIT = 500_000
RARE_BRAND_MIN_COUNT = 10
TWO_WORD_BRANDS = {"Land Rover", "Ashok Leyland"}
OWNER_MAP = {
    "First Owner": 1, "Second Owner": 2, "Third Owner": 3,
    "Fourth & Above Owner": 4, "Test Drive Car": 5,
}


def extract_brand(name):
    words = str(name).split()
    leading_pair = " ".join(words[:2])
    return leading_pair if leading_pair in TWO_WORD_BRANDS else words[0]


def load_clean_data(path="../datasets/Cars.csv", verbose=True):
    df = pd.read_csv(path)
    raw_rows = len(df)
    df = df.drop_duplicates().copy()
    df = df.drop(columns=["torque"])
    df = df[df["owner"] != "Test Drive Car"].copy()
    df["owner"] = df["owner"].map(OWNER_MAP)
    df = df[~df["fuel"].isin(["CNG", "LPG"])].copy()
    for column in ["mileage", "engine", "max_power"]:
        df[column] = pd.to_numeric(
            df[column].astype("string").str.extract(r"([-+]?\d*\.?\d+)", expand=False),
            errors="coerce",
        )
        df.loc[df[column] == 0, column] = np.nan
    df = df[df["km_driven"] <= KM_DRIVEN_LIMIT].copy()
    df["brand"] = df["name"].map(extract_brand)
    df = df.drop(columns=["name"])
    counts = df["brand"].value_counts()
    df["brand"] = df["brand"].where(df["brand"].map(counts) >= RARE_BRAND_MIN_COUNT, "Other")
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[["brand"] + [column for column in df.columns if column != "brand"]]
    if verbose:
        print(f"raw rows: {raw_rows} -> cleaned rows: {len(df)}")
    return df


def price_bin_edges(training_prices):
    """Quartiles learned from training data only; infinities cover future extremes."""
    quantiles = np.quantile(np.asarray(training_prices, dtype=float), [0.25, 0.50, 0.75])
    if np.unique(quantiles).size != 3:
        raise ValueError("price quartiles are not distinct")
    return np.r_[-np.inf, quantiles, np.inf]


def bucket_prices(prices, edges):
    return pd.cut(prices, bins=edges, labels=[0, 1, 2, 3], include_lowest=True).astype(int)


def normalise_missing(input_row):
    row = input_row.copy()
    for column in row.columns:
        row[column] = row[column].where(row[column].notna(), np.nan)
    return row
