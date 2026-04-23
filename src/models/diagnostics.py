"""Diagnostic utilities for OLS stability checks and plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import recursive_olsresiduals


def fit_ols_for_diagnostics(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    """Fit a standard OLS model for downstream diagnostic plots."""
    model_df = df[[y_col] + x_cols].copy()
    for col in [y_col] + x_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna().reset_index(drop=True)
    X = sm.add_constant(model_df[x_cols], has_constant="add")
    y = model_df[y_col]
    return sm.OLS(y, X).fit(), model_df


def cusum_dataframe(model) -> pd.DataFrame:
    """Build a dataframe containing recursive residual CUSUM series and confidence bands."""
    skip_n = max(3, int(model.model.exog.shape[1]) + 1)
    out = recursive_olsresiduals(model, skip=skip_n)
    cusum = out[5]
    confint = out[6].T
    min_len = min(len(cusum), len(confint))
    cusum = cusum[-min_len:]
    confint = confint[-min_len:]
    return pd.DataFrame({
        "step": range(min_len),
        "cusum": cusum,
        "lower_5pct": confint[:, 0],
        "upper_5pct": confint[:, 1],
    })


def plot_cusum(cusum_df: pd.DataFrame, output_file: Path) -> None:
    """Save a CUSUM stability plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cusum_df["step"], cusum_df["cusum"], label="CUSUM")
    ax.plot(cusum_df["step"], cusum_df["lower_5pct"], linestyle="--", label="5% bound")
    ax.plot(cusum_df["step"], cusum_df["upper_5pct"], linestyle="--")
    ax.set_title("CUSUM Stability Test")
    ax.set_xlabel("Recursive step")
    ax.set_ylabel("CUSUM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
