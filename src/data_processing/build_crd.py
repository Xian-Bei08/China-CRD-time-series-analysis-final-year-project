"""Functions for extracting the national CRD outcome series used as the main dependent variable."""
import pandas as pd


def build_crd_main(
    crd: pd.DataFrame,
    location: str,
    sex: str,
    age_name: str,
    metric: str,
    crd_measure: str,
    crd_cause: str,
) -> pd.DataFrame:
    """
    Build the annual national CRD DALY rate series used as the dependent variable.

    The function filters the raw GBD-style table down to the exact analytical
    definition required by the dissertation, then keeps only the year and value
    columns needed for downstream merging.

    Output: year, crd_daly_rate
    """
    # Filter the raw table to the target location, demographic group, metric,
    # measure, and cause definition used in the main analysis.
    d = crd[
        (crd["location_name"] == location) &
        (crd["sex_name"] == sex) &
        (crd["age_name"] == age_name) &
        (crd["metric_name"] == metric) &
        (crd["measure_name"] == crd_measure) &
        (crd["cause_name"] == crd_cause)
    ].copy()

    # Keep a clean two-column annual series for the later merge step.
    d = (
        d[["year", "val"]]
        .rename(columns={"val": "crd_daly_rate"})
        .sort_values("year")
        .drop_duplicates(subset=["year"])
    )

    return d