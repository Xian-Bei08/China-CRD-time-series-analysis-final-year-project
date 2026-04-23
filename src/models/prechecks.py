"""Pre-modelling diagnostic utilities for missingness, stationarity, correlation, and VIF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages for each column."""
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
    }).reset_index()
    return missing.rename(columns={"index": "variable"})


def descriptive_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return concise descriptive statistics for selected numeric columns."""
    out = df[columns].apply(pd.to_numeric, errors="coerce")
    desc = out.describe().T
    desc["skew"] = out.skew()
    desc["kurtosis"] = out.kurtosis()
    return desc.reset_index().rename(columns={"index": "variable"})


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a Pearson correlation matrix for selected columns."""
    out = df[columns].apply(pd.to_numeric, errors="coerce")
    return out.corr()


def plot_correlation_heatmap(corr_df: pd.DataFrame, output_file: Path) -> None:
    """Save a simple correlation heatmap using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr_df, aspect="auto")
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=90)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def standardize_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Z-score standardise selected predictors."""
    X = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return scaled.dropna(axis=1, how="any")


def calculate_vif(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Calculate VIF values for selected predictors."""
    X = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if X.shape[1] < 2:
        return pd.DataFrame({"variable": X.columns, "vif": [np.nan] * X.shape[1]})
    vif_data = pd.DataFrame({
        "variable": X.columns,
        "vif": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    return vif_data.sort_values("vif", ascending=False).reset_index(drop=True)


def vif_feature_selection(
    df: pd.DataFrame,
    columns: list[str],
    threshold: float = 10.0,
    scale_first: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Iteratively remove the predictor with the highest VIF until all are below threshold."""
    X = standardize_features(df, columns) if scale_first else df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    selected = list(X.columns)
    initial_vif_df = calculate_vif(X, selected)
    history_rows = []
    step = 0

    while len(selected) >= 2:
        current_vif = calculate_vif(X, selected)
        max_row = current_vif.iloc[0]
        max_vif = float(max_row["vif"])
        max_var = str(max_row["variable"])

        history_rows.append({
            "step": step,
            "action": "check" if max_vif <= threshold else "remove",
            "variable": max_var,
            "vif": max_vif,
            "n_remaining_before_action": len(selected),
            "remaining_variables": ", ".join(selected),
        })

        if max_vif <= threshold:
            break

        selected.remove(max_var)
        step += 1

    return initial_vif_df, pd.DataFrame(history_rows), selected


def run_adf_on_series(series: pd.Series, name: str, regression: str = "c") -> dict:
    """Run an ADF test on a single time series."""
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) < 8 or s.nunique() <= 1:
        return {
            "variable": name,
            "regression": regression,
            "adf_statistic": np.nan,
            "p_value": np.nan,
            "used_lag": np.nan,
            "n_obs": len(s),
            "critical_1pct": np.nan,
            "critical_5pct": np.nan,
            "critical_10pct": np.nan,
        }

    try:
        result = adfuller(s, regression=regression, autolag="AIC")
        return {
            "variable": name,
            "regression": regression,
            "adf_statistic": result[0],
            "p_value": result[1],
            "used_lag": result[2],
            "n_obs": result[3],
            "critical_1pct": result[4].get("1%"),
            "critical_5pct": result[4].get("5%"),
            "critical_10pct": result[4].get("10%"),
        }
    except Exception:
        return {
            "variable": name,
            "regression": regression,
            "adf_statistic": np.nan,
            "p_value": np.nan,
            "used_lag": np.nan,
            "n_obs": len(s),
            "critical_1pct": np.nan,
            "critical_5pct": np.nan,
            "critical_10pct": np.nan,
        }


def run_adf_tests(df: pd.DataFrame, columns: list[str], regression: str = "c") -> pd.DataFrame:
    """Run ADF tests across a list of columns."""
    return pd.DataFrame([run_adf_on_series(df[col], col, regression=regression) for col in columns])


def infer_integration_order(df: pd.DataFrame, columns: list[str], regression: str = "c") -> pd.DataFrame:
    """Infer a rough integration order by comparing level and first-difference ADF results."""
    rows = []
    for col in columns:
        level_res = run_adf_on_series(df[col], col, regression=regression)
        diff_res = run_adf_on_series(pd.to_numeric(df[col], errors="coerce").diff(), f"d_{col}", regression=regression)

        if pd.notna(level_res["p_value"]) and level_res["p_value"] < 0.05:
            inferred = "I(0)"
        elif pd.notna(diff_res["p_value"]) and diff_res["p_value"] < 0.05:
            inferred = "I(1)"
        else:
            inferred = "Inconclusive (ADF low power)"

        rows.append({
            "variable": col,
            "level_p_value": level_res["p_value"],
            "diff_p_value": diff_res["p_value"],
            "inferred_order": inferred,
        })

    return pd.DataFrame(rows)


def run_targeted_adf_rechecks(df: pd.DataFrame, columns: list[str], regression: str = "ct") -> pd.DataFrame:
    """Run ADF rechecks with trend for selected variables."""
    return pd.DataFrame([run_adf_on_series(df[col], col, regression=regression) for col in columns])
