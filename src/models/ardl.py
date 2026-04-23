"""ARDL modelling helpers, including lag-order selection, coefficient extraction, and fitted-value exports."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.ardl import ardl_select_order


def prepare_ardl_data(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Prepare a clean ARDL input dataset.
    """
    required_cols = [year_col, y_col] + x_cols

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for ARDL: {missing_cols}")

    model_df = df[required_cols].copy()
    model_df = model_df.sort_values(year_col).reset_index(drop=True)

    for col in [y_col] + x_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df = model_df.dropna(subset=[y_col] + x_cols).reset_index(drop=True)
    return model_df


def fit_ardl(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    maxlag: int = 1,
    maxorder: int | dict[str, int] = 1,
    ic: str = "bic",
    trend: str = "c",
    year_col: str = "year",
    causal: bool = False,
):
    """
    Select and fit an ARDL model using information criteria.
    """
    model_df = prepare_ardl_data(df=df, y_col=y_col, x_cols=x_cols, year_col=year_col)

    hold_back = maxlag
    if isinstance(maxorder, int):
        hold_back = max(hold_back, maxorder)
    elif isinstance(maxorder, dict):
        hold_back = max(hold_back, max(maxorder.values()))

    selector = ardl_select_order(
        endog=model_df[y_col],
        maxlag=maxlag,
        exog=model_df[x_cols],
        maxorder=maxorder,
        ic=ic,
        trend=trend,
        causal=causal,
        hold_back=hold_back,
        missing="drop",
    )
    fitted_model = selector.model.fit()
    return model_df, selector, fitted_model


def selection_table(selector, ic: str = "bic", top_n: int = 10) -> pd.DataFrame:
    """
    Convert the ARDL selection results into a readable ranking table.
    """
    ic_series = getattr(selector, ic).head(top_n)

    rows = []
    for rank, (score, spec) in enumerate(ic_series.items(), start=1):
        ar_lags, dl_lags = spec
        rows.append({
            "rank": rank,
            "criterion": ic.upper(),
            "score": score,
            "ar_lags": str(ar_lags),
            "dl_lags": str(dl_lags),
        })

    return pd.DataFrame(rows)


def extract_ardl_coefficients(fitted_model) -> pd.DataFrame:
    """
    Extract coefficient estimates from a fitted ARDL model.
    """
    ci = fitted_model.conf_int()
    coef_df = pd.DataFrame({
        "variable": fitted_model.params.index,
        "coefficient": fitted_model.params.values,
        "std_error": fitted_model.bse.values,
        "t_value": fitted_model.tvalues.values,
        "p_value": fitted_model.pvalues.values,
        "ci_lower": ci[0].values,
        "ci_upper": ci[1].values,
    })
    return coef_df


def extract_ardl_model_fit(fitted_model) -> pd.DataFrame:
    """
    Extract a concise set of model fit statistics for reporting.
    """
    fit_df = pd.DataFrame({
        "metric": ["n_obs", "aic", "bic", "hqic", "loglik", "sigma2"],
        "value": [
            fitted_model.nobs,
            fitted_model.aic,
            fitted_model.bic,
            fitted_model.hqic,
            fitted_model.llf,
            fitted_model.sigma2,
        ],
    })
    return fit_df


def ardl_fitted_export(model_df: pd.DataFrame, fitted_model, y_col: str, year_col: str = "year") -> pd.DataFrame:
    """
    Export actual, fitted, and residual series aligned to the ARDL estimation sample.
    """
    fitted_values = pd.Series(fitted_model.fittedvalues)
    fitted_df = pd.DataFrame({
        year_col: model_df.loc[fitted_values.index, year_col].values,
        y_col: model_df.loc[fitted_values.index, y_col].values,
        "fitted": fitted_values.values,
    })
    fitted_df["residual"] = fitted_df[y_col] - fitted_df["fitted"]
    return fitted_df.reset_index(drop=True)


def plot_ardl_actual_vs_fitted(
    fitted_df: pd.DataFrame,
    y_col: str,
    output_file: Path,
    year_col: str = "year",
) -> None:
    """
    Plot actual and fitted values for the ARDL model.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fitted_df[year_col], fitted_df[y_col], label="Actual")
    ax.plot(fitted_df[year_col], fitted_df["fitted"], label="Fitted")
    ax.set_title("ARDL: Actual vs Fitted")
    ax.set_xlabel("Year")
    ax.set_ylabel(y_col)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def selected_order_text(selector) -> str:
    """
    Build a human-readable summary of the selected ARDL order.
    """
    model = selector.model
    lines = [
        f"Selected model: {model}",
        f"AR lags: {model.ar_lags}",
        f"Distributed lags: {model.dl_lags}",
        f"Trend: {model.trend}",
        f"Seasonal: {model.seasonal}",
        f"Period: {model.period}",
    ]
    return "\n".join(lines)
