"""Bounds test and ECM utilities built on top of the selected ARDL specification."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.api import UECM

from src.models.ardl import fit_ardl


def fit_uecm_from_ardl(selector):
    """
    Attempt to construct a UECM from the selected ARDL model.

    This will fail if the selected ARDL specification is not compatible with
    the UECM representation required by statsmodels.
    """
    uecm_model = UECM.from_ardl(selector.model)
    uecm_res = uecm_model.fit()
    return uecm_model, uecm_res



def fit_manual_uecm(
    model_df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    lags: int = 1,
    exog_order: int = 1,
    trend: str = "c",
):
    """
    Fit a theory-guided fallback UECM when UECM.from_ardl fails.

    The fallback imposes a regular UECM(1,1,...,1) structure so that the
    bounds test can still be performed on a valid error-correction form.
    """
    y = model_df[y_col]
    X = model_df[x_cols]
    order = {col: exog_order for col in x_cols}

    uecm_model = UECM(
        endog=y,
        lags=lags,
        exog=X,
        order=order,
        trend=trend,
    )
    uecm_res = uecm_model.fit()
    return uecm_model, uecm_res



def run_bounds_test(uecm_res, case: int = 3):
    """
    Run the Pesaran-Shin-Smith bounds test on a fitted UECM result.
    """
    return uecm_res.bounds_test(case=case)



def bounds_result_table(bounds_res) -> pd.DataFrame:
    """
    Convert bounds test output into a concise dissertation-friendly table.
    """
    p_values = getattr(bounds_res, "p_values", None)
    lower_p = p_values.get("lower") if p_values is not None else None
    upper_p = p_values.get("upper") if p_values is not None else None

    return pd.DataFrame(
        {
            "metric": ["f_stat", "lower_p_value", "upper_p_value"],
            "value": [bounds_res.stat, lower_p, upper_p],
        }
    )



def uecm_coefficients_table(uecm_res) -> pd.DataFrame:
    """
    Extract coefficient estimates from a fitted UECM model.
    """
    ci = uecm_res.conf_int()
    return pd.DataFrame(
        {
            "variable": uecm_res.params.index,
            "coefficient": uecm_res.params.values,
            "std_error": uecm_res.bse.values,
            "t_value": uecm_res.tvalues.values,
            "p_value": uecm_res.pvalues.values,
            "ci_lower": ci[0].values,
            "ci_upper": ci[1].values,
        }
    )



def ci_table(uecm_res) -> pd.DataFrame:
    """
    Extract the cointegrating vector as a table when available.
    """
    ci_summary = uecm_res.ci_summary()
    table = ci_summary.tables[0]
    return pd.DataFrame(table.data[1:], columns=table.data[0])



def ecm_speed_of_adjustment_table(uecm_res) -> pd.DataFrame:
    """
    Extract the lagged dependent level term, usually interpreted as the
    error-correction / speed-of-adjustment coefficient.
    """
    coef_df = uecm_coefficients_table(uecm_res)
    mask = coef_df["variable"].str.contains(rf"\.L1$", regex=True) & coef_df["variable"].str.startswith("crd_daly_rate")
    return coef_df.loc[mask].reset_index(drop=True)



def save_text(text: str, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)



def bounds_interpretation_text(bounds_res, case: int = 3) -> str:
    """
    Create a plain-language note for the bounds test result.
    """
    crit = bounds_res.crit_vals
    stat = bounds_res.stat
    upper_95 = crit.loc[95.0, "upper"]
    lower_95 = crit.loc[95.0, "lower"]

    if stat > upper_95:
        verdict = "Evidence supports a long-run level relationship at the 5% level."
    elif stat < lower_95:
        verdict = "No evidence of a long-run level relationship at the 5% level."
    else:
        verdict = "The result is inconclusive at the 5% level."

    return (
        f"Bounds test case: {case}\n"
        f"F-statistic: {stat:.6f}\n"
        f"5% lower bound: {lower_95:.6f}\n"
        f"5% upper bound: {upper_95:.6f}\n"
        f"Interpretation: {verdict}\n"
    )



def fitted_differences_export(model_df: pd.DataFrame, uecm_res, y_col: str, year_col: str = "year") -> pd.DataFrame:
    """
    Export actual and fitted first differences from the UECM model.
    """
    fitted_values = pd.Series(uecm_res.fittedvalues)
    diff_actual = model_df[y_col].diff()

    fitted_df = pd.DataFrame(
        {
            year_col: model_df.loc[fitted_values.index, year_col].values,
            f"d_{y_col}": diff_actual.loc[fitted_values.index].values,
            "fitted": fitted_values.values,
        }
    )
    fitted_df["residual"] = fitted_df[f"d_{y_col}"] - fitted_df["fitted"]
    return fitted_df.reset_index(drop=True)



def plot_uecm_actual_vs_fitted(
    fitted_df: pd.DataFrame,
    y_col: str,
    output_file: Path,
    year_col: str = "year",
) -> None:
    """
    Plot actual vs fitted first differences for the UECM model.
    """
    diff_col = f"d_{y_col}"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fitted_df[year_col], fitted_df[diff_col], label="Actual ΔY")
    ax.plot(fitted_df[year_col], fitted_df["fitted"], label="Fitted ΔY")
    ax.set_title("UECM: Actual vs Fitted First Difference")
    ax.set_xlabel("Year")
    ax.set_ylabel(diff_col)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()



def fit_ardl_then_bounds(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    maxlag: int = 1,
    maxorder: int | dict[str, int] = 1,
    ic: str = "bic",
    trend: str = "c",
    year_col: str = "year",
    causal: bool = False,
    bounds_case: int = 3,
):
    """
    Convenience wrapper for the complete ARDL -> UECM -> bounds workflow.
    """
    model_df, selector, ardl_res = fit_ardl(
        df=df,
        y_col=y_col,
        x_cols=x_cols,
        maxlag=maxlag,
        maxorder=maxorder,
        ic=ic,
        trend=trend,
        year_col=year_col,
        causal=causal,
    )

    used_manual_fallback = False
    fallback_reason = ""

    try:
        _, uecm_res = fit_uecm_from_ardl(selector)
    except Exception as e:
        used_manual_fallback = True
        fallback_reason = str(e)
        _, uecm_res = fit_manual_uecm(
            model_df=model_df,
            y_col=y_col,
            x_cols=x_cols,
            lags=maxlag,
            exog_order=1,
            trend=trend,
        )

    bounds_res = run_bounds_test(uecm_res, case=bounds_case)

    return {
        "model_df": model_df,
        "selector": selector,
        "ardl_res": ardl_res,
        "uecm_res": uecm_res,
        "bounds_res": bounds_res,
        "used_manual_fallback": used_manual_fallback,
        "fallback_reason": fallback_reason,
    }
