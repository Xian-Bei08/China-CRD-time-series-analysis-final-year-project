"""Functions for reshaping ageing and government health expenditure data into annual analytical series."""

import re
import numpy as np
import pandas as pd


def _linear_trend_impute_nonnegative(series: pd.Series, years: pd.Series) -> pd.Series:
    """
    Fill interior gaps by linear interpolation and fill edge gaps by fitting a
    simple linear trend to the observed part of the series.

    After imputation, enforce a non-negative constraint:
    if early extrapolated values are negative, replace them with the first
    non-negative imputed value.
    """
    out = pd.to_numeric(series, errors="coerce").copy()
    year_values = pd.to_numeric(years, errors="coerce")

    # Fill interior gaps only.
    out = out.interpolate(method="linear")

    observed = out.notna()
    if observed.sum() < 2:
        return out

    # Fit linear trend on observed values.
    x = year_values.loc[observed].to_numpy(dtype=float)
    y = out.loc[observed].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)

    # Fill remaining missing values by trend extrapolation.
    missing = out.isna()
    if missing.any():
        out.loc[missing] = slope * year_values.loc[missing] + intercept

    # Enforce non-negative values.
    non_negative_mask = out >= 0
    if non_negative_mask.any():
        first_valid_non_negative = out.loc[non_negative_mask].iloc[0]
        out.loc[out < 0] = first_valid_non_negative

    return out


def build_gdp_ageing_health_main(
    gdp_ageing: pd.DataFrame,
    health_exp: pd.DataFrame,
    start_year: int,
    end_year: int,
    impute_health_exp: bool = True,
) -> pd.DataFrame:
    """
    Build AGEING + health policy table.

    Output columns
    --------------
    year, ageing_65_plus, gov_health_exp_pct_gdp,
    gov_health_exp_pct_gdp_observed, gov_health_exp_pct_gdp_imputed

    Notes
    -----
    - Keeps the requested year range even when some years are missing.
    - Government health expenditure can be imputed backwards to 1990 using a
      simple linear-trend extrapolation over the observed annual series.
    """

    # -------------------------
    # Step 1. Reshape the ageing indicator.
    # -------------------------
    series_map = {
        "Population ages 65 and above (% of total population)": "ageing_65_plus"
    }

    d = gdp_ageing[gdp_ageing["Series Name"].isin(series_map.keys())].copy()
    year_cols = [c for c in d.columns if re.match(r"^\d{4}\s\[YR\d{4}\]$", str(c))]

    ageing_long = d.melt(
        id_vars=["Country Name", "Series Name"],
        value_vars=year_cols,
        var_name="year_raw",
        value_name="value",
    )

    ageing_long["year"] = ageing_long["year_raw"].str.extract(r"^(\d{4})").astype(int)
    ageing_long["value"] = pd.to_numeric(ageing_long["value"], errors="coerce")

    ageing_wide = (
        ageing_long.pivot_table(
            index="year",
            columns="Series Name",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns=series_map)
    )

    # -------------------------
    # Step 2. Reshape the government health expenditure indicator.
    # -------------------------
    h = health_exp.copy()
    year_cols_h = [c for c in h.columns if re.match(r"^\d{4}\s\[YR\d{4}\]$", str(c))]

    health_long = h.melt(
        id_vars=["Country Name", "Series Name"],
        value_vars=year_cols_h,
        var_name="year_raw",
        value_name="gov_health_exp_pct_gdp",
    )

    health_long["year"] = health_long["year_raw"].str.extract(r"^(\d{4})").astype(int)
    health_long["gov_health_exp_pct_gdp"] = pd.to_numeric(
        health_long["gov_health_exp_pct_gdp"],
        errors="coerce",
    )

    health_wide = health_long.groupby("year", as_index=False)["gov_health_exp_pct_gdp"].mean()
    health_wide["gov_health_exp_pct_gdp_observed"] = health_wide["gov_health_exp_pct_gdp"]

    # -------------------------
    # Step 3. Build the full year skeleton and optionally impute health expenditure.
    # -------------------------
    skeleton = pd.DataFrame({"year": range(start_year, end_year + 1)})

    final = (
        skeleton
        .merge(ageing_wide, on="year", how="left")
        .merge(health_wide, on="year", how="left")
        .sort_values("year")
        .reset_index(drop=True)
    )

    if impute_health_exp:
        observed_before = final["gov_health_exp_pct_gdp"].notna()
        final["gov_health_exp_pct_gdp"] = _linear_trend_impute_nonnegative(
            series=final["gov_health_exp_pct_gdp"],
            years=final["year"],
        )
        final["gov_health_exp_pct_gdp_imputed"] = (
            final["gov_health_exp_pct_gdp"].notna() & ~observed_before
        )
    else:
        final["gov_health_exp_pct_gdp_imputed"] = False

    return final
