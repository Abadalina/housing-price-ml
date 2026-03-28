"""
Feature engineering for the Airbnb Madrid price prediction model.
Author: Alejandro Abadal
"""

import pandas as pd
import numpy as np


# ── Columns used ─────────────────────────────────────────────────────────────
RAW_COLS = [
    'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
    'latitude', 'longitude', 'room_type', 'accommodates',
    'bedrooms', 'beds', 'price', 'minimum_nights',
    'number_of_reviews', 'review_scores_rating', 'reviews_per_month',
    'host_is_superhost', 'host_listings_count',
    'instant_bookable', 'availability_365', 'bathrooms_text'
]

TARGET = 'price'

CATEGORICAL_COLS = ['neighbourhood_cleansed', 'room_type']

NUMERIC_COLS = [
    'accommodates', 'bedrooms', 'beds', 'bathrooms',
    'minimum_nights', 'number_of_reviews', 'review_scores_rating',
    'reviews_per_month', 'host_listings_count', 'availability_365',
    'latitude', 'longitude'
]

BINARY_COLS = ['host_is_superhost', 'instant_bookable']


# ── Cleaning ──────────────────────────────────────────────────────────────────
def clean_price(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )


def extract_bathrooms(series: pd.Series) -> pd.Series:
    """Extract numeric value from 'bathrooms_text' column (e.g. '1.5 baths' -> 1.5)."""
    return (
        series.astype(str)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw CSV and return a clean dataframe ready for feature engineering."""
    df = pd.read_csv(path, low_memory=False)

    # Keep only relevant columns that exist
    cols = [c for c in RAW_COLS if c in df.columns]
    df = df[cols].copy()

    # Price
    if not pd.api.types.is_numeric_dtype(df['price']):
        df['price'] = clean_price(df['price'])

    # Bathrooms
    if 'bathrooms_text' in df.columns:
        df['bathrooms'] = extract_bathrooms(df['bathrooms_text'])
        df = df.drop(columns=['bathrooms_text'])

    # Boolean columns
    for col in ['host_is_superhost', 'instant_bookable']:
        if col in df.columns:
            df[col] = df[col].map({'t': 1, 'f': 0, True: 1, False: 0})

    # Drop rows with no price or no target features
    df = df.dropna(subset=['price', 'accommodates', 'room_type'])
    df = df[df['price'] > 0]

    # Remove price outliers (1st–99th percentile)
    q_low  = df['price'].quantile(0.01)
    q_high = df['price'].quantile(0.99)
    df = df[df['price'].between(q_low, q_high)]

    # Remove listings with absurd minimum nights
    if 'minimum_nights' in df.columns:
        df = df[df['minimum_nights'] <= 365]

    return df.reset_index(drop=True)


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the dataframe."""
    df = df.copy()

    # Beds per person
    if 'beds' in df.columns:
        df['beds_per_person'] = df['beds'] / df['accommodates'].clip(lower=1)

    # Has reviews flag
    df['has_reviews'] = (df['number_of_reviews'].fillna(0) > 0).astype(int)

    # High availability flag (>270 days/year)
    if 'availability_365' in df.columns:
        df['high_availability'] = (df['availability_365'] > 270).astype(int)

    # Neighbourhood median price (target encoding — computed on full dataset)
    nb_median = df.groupby('neighbourhood_cleansed')['price'].transform('median')
    df['neighbourhood_median_price'] = nb_median

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return the list of feature columns to use for training."""
    base = [c for c in NUMERIC_COLS if c in df.columns]
    base += [c for c in BINARY_COLS if c in df.columns]
    base += [c for c in CATEGORICAL_COLS if c in df.columns]
    extra = [c for c in ['beds_per_person', 'has_reviews',
                          'high_availability', 'neighbourhood_median_price']
             if c in df.columns]
    return base + extra
