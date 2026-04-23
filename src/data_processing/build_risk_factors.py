"""Functions for building annual national exposure series from PM2.5, ozone, and household air pollution data."""

import pandas as pd

def build_pm(pm: pd.DataFrame, location_id: int) -> pd.DataFrame:
    """
    Build the annual PM2.5 exposure series for the selected national location.
    Output: year, pm25
    """
    # Restrict the raw table to the requested country-level identifier.
    d = pm[pm["location_id"] == location_id].copy()

    # Keep only the annual mean exposure used in the merged analysis dataset.
    d = (
        d[["year_id", "mean"]]
        .rename(columns={"year_id": "year", "mean": "pm25"})
        .sort_values("year")
        .drop_duplicates(subset=["year"])
    )

    return d


def build_ozone(ozone: pd.DataFrame, location_id: int) -> pd.DataFrame:
    """
    Build ozone exposure series.
    Output: year, ozone
    """
    # Restrict the raw table to the requested country-level identifier.
    d = ozone[ozone["location_id"] == location_id].copy()

    # Keep only the annual mean exposure used in the merged analysis dataset.
    d = (
        d[["year_id", "mean"]]
        .rename(columns={"year_id": "year", "mean": "ozone"})
        .sort_values("year")
        .drop_duplicates(subset=["year"])
    )

    return d


def build_household_pm(hap: pd.DataFrame, location_id: int) -> pd.DataFrame:
    """
    Build household air pollution series.
    Since this file only has Male/Female, average them into one yearly national value.
    Output: year, household_pm
    """
    # Restrict the raw table to the requested country-level identifier.
    d = hap[hap["location_id"] == location_id].copy()

    # Collapse sex-specific rows into one yearly mean for the analytical dataset.
    d = (
        d.groupby("year_id", as_index=False)["mean"]
        .mean()
        .rename(columns={"year_id": "year", "mean": "household_pm"})
        .sort_values("year")
    )

    return d