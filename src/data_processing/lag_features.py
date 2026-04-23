"""Feature-engineering helpers for lag construction and first-difference modelling datasets."""

from __future__ import annotations

import pandas as pd


def add_trend_feature(
    df: pd.DataFrame,
    year_col: str = "year",
    trend_col: str = "trend",
) -> pd.DataFrame:
    """Add a deterministic linear time trend that starts at 1."""
    out = df.copy().sort_values(year_col).reset_index(drop=True)
    out[trend_col] = range(1, len(out) + 1)
    return out


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
    year_col: str = "year",
) -> pd.DataFrame:
    """Add lagged versions of selected variables to a time-ordered dataframe."""
    out = df.copy().sort_values(year_col).reset_index(drop=True)

    for col in columns:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)

    return out


def first_difference_columns(
    df: pd.DataFrame,
    columns: list[str],
    year_col: str = "year",
    prefix: str = "d_",
) -> pd.DataFrame:
    """Create first-differenced versions of selected columns."""
    out = df.copy().sort_values(year_col).reset_index(drop=True)

    for col in columns:
        out[f"{prefix}{col}"] = pd.to_numeric(out[col], errors="coerce").diff()

    return out


def build_feature_dataset(
    df: pd.DataFrame,
    lag_columns: list[str],
    lags: list[int],
    drop_na: bool = True,
    year_col: str = "year",
    add_trend: bool = True,
) -> pd.DataFrame:
    """Build the lag-expanded modelling dataset from the level analysis table."""
    out = df.copy().sort_values(year_col).reset_index(drop=False)

    if add_trend:
        out = add_trend_feature(out, year_col=year_col, trend_col="trend")

    out = add_lag_features(
        out,
        columns=lag_columns,
        lags=lags,
        year_col=year_col,
    )

    if drop_na:
        out = out.dropna().reset_index(drop=True)

    return out


def build_differenced_modelling_dataset(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    lags: list[int],
    year_col: str = "year",
    add_trend: bool = True,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Build the dissertation modelling dataset using first differences and lagged
    differenced predictors.

    Output columns include:
    year, trend, d_y, d_x_lag0, d_x_lag1, d_x_lag2, and the original level
    columns preserved for downstream ARDL-ECM work.
    """
    work = df.copy().sort_values(year_col).reset_index(drop=True)

    if add_trend:
        work = add_trend_feature(work, year_col=year_col, trend_col="trend")

    work = first_difference_columns(work, [y_col] + x_cols, year_col=year_col, prefix="d_")

    diff_cols = [f"d_{col}" for col in x_cols]
    positive_lags = [lag for lag in lags if lag > 0]
    if positive_lags:
        work = add_lag_features(work, diff_cols, lags=positive_lags, year_col=year_col)

    rename_map = {f"d_{col}": f"d_{col}_lag0" for col in x_cols}
    work = work.rename(columns=rename_map)

    if drop_na:
        required_cols = [f"d_{y_col}"] + [f"d_{col}_lag{lag}" for col in x_cols for lag in lags]
        work = work.dropna(subset=required_cols).reset_index(drop=True)

    return work