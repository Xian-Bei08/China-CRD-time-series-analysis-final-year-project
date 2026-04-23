import pandas as pd

def merge_main_dataset(
    crd_main: pd.DataFrame,
    pm_main: pd.DataFrame,
    ozone_main: pd.DataFrame,
    hap_main: pd.DataFrame,
    ageing_main: pd.DataFrame,
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    """
    Merge all main datasets into one final analysis dataset.

    Keeps full year range from start_year to end_year.
    Missing values are preserved as NaN.
    """

    # -------------------------
    # Full year skeleton
    # -------------------------
    skeleton = pd.DataFrame({
        "year": range(start_year, end_year + 1)
    })

    # -------------------------
    # Sequential left joins
    # -------------------------
    final = (
        skeleton
        .merge(crd_main, on="year", how="left")
        .merge(pm_main, on="year", how="left")
        .merge(ozone_main, on="year", how="left")
        .merge(hap_main, on="year", how="left")
        .merge(ageing_main, on="year", how="left")
        .sort_values("year")
        .reset_index(drop=True)
    )

    return final