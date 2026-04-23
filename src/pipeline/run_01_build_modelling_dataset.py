"""Build the new dissertation-ready modelling datasets for differenced screening and level-series ARDL work."""

from __future__ import annotations

from src.config import (
    CRD_FILE,
    PM25_FILE,
    OZONE_FILE,
    HAP_FILE,
    AGEING_FILE,
    GOV_HEALTH_EXP_FILE,
    CRD_MAIN_FILE,
    PM25_MAIN_FILE,
    OZONE_MAIN_FILE,
    HOUSEHOLD_PM_MAIN_FILE,
    AGEING_HEALTH_MAIN_FILE,
    FINAL_ANALYSIS_FILE,
    MODELLING_DIFF_FILE,
    MODELLING_LEVEL_FILE,
    DATA_INTERIM,
    DATA_PROCESSED,
    LOCATION,
    LOCATION_ID,
    SEX,
    AGE_NAME,
    METRIC,
    CRD_MEASURE,
    CRD_CAUSE,
    START_YEAR,
    END_YEAR,
    BASE_LEVEL_X_COLS,
)
from src.loaders import (
    load_crd_data,
    load_pm25_data,
    load_ozone_data,
    load_hap_data,
    load_gdp_ageing_data,
    load_health_exp_data,
)
from src.data_processing.build_crd import build_crd_main
from src.data_processing.build_risk_factors import build_pm, build_ozone, build_household_pm
from src.data_processing.build_macro_vars import build_gdp_ageing_health_main
from src.data_processing.merge_main_dataset import merge_main_dataset
from src.data_processing.lag_features import build_differenced_modelling_dataset, add_trend_feature


def main() -> None:
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    crd = load_crd_data(CRD_FILE)
    pm = load_pm25_data(PM25_FILE)
    ozone = load_ozone_data(OZONE_FILE)
    hap = load_hap_data(HAP_FILE)
    ageing = load_gdp_ageing_data(AGEING_FILE)
    gov_health = load_health_exp_data(GOV_HEALTH_EXP_FILE)

    crd_main = build_crd_main(
        crd=crd,
        location=LOCATION,
        sex=SEX,
        age_name=AGE_NAME,
        metric=METRIC,
        crd_measure=CRD_MEASURE,
        crd_cause=CRD_CAUSE,
    )
    pm_main = build_pm(pm, LOCATION_ID)
    ozone_main = build_ozone(ozone, LOCATION_ID)
    hap_main = build_household_pm(hap, LOCATION_ID)
    ageing_health_main = build_gdp_ageing_health_main(
        gdp_ageing=ageing,
        health_exp=gov_health,
        start_year=START_YEAR,
        end_year=END_YEAR,
        impute_health_exp=True,
    )

    crd_main.to_csv(CRD_MAIN_FILE, index=False)
    pm_main.to_csv(PM25_MAIN_FILE, index=False)
    ozone_main.to_csv(OZONE_MAIN_FILE, index=False)
    hap_main.to_csv(HOUSEHOLD_PM_MAIN_FILE, index=False)
    ageing_health_main.to_csv(AGEING_HEALTH_MAIN_FILE, index=False)

    final_df = merge_main_dataset(
        crd_main=crd_main,
        pm_main=pm_main,
        ozone_main=ozone_main,
        hap_main=hap_main,
        ageing_main=ageing_health_main,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )
    final_df.to_csv(FINAL_ANALYSIS_FILE, index=False)

    level_df = add_trend_feature(
        final_df,
        year_col="year",
        trend_col="trend",
    )
    level_df.to_csv(MODELLING_LEVEL_FILE, index=False)

    diff_df = build_differenced_modelling_dataset(
        df=final_df,
        y_col="crd_daly_rate",
        x_cols=BASE_LEVEL_X_COLS,
        lags=[0, 1, 2],
        year_col="year",
        add_trend=True,
        drop_na=True,
    )
    diff_df.to_csv(MODELLING_DIFF_FILE, index=False)

    print(f"Saved level modelling dataset to: {MODELLING_LEVEL_FILE}")
    print(f"Saved differenced modelling dataset to: {MODELLING_DIFF_FILE}")
    print("\n=== Differenced modelling dataset preview ===")
    print(diff_df.head())


if __name__ == "__main__":
    main()
