"""Run dissertation prechecks on the differenced modelling dataset and level integration order screen."""

from __future__ import annotations

import pandas as pd

from src.config import (
    MODELLING_DIFF_FILE,
    MODELLING_LEVEL_FILE,
    PRECHECK_DIR,
    Y_COL,
    BASE_LEVEL_X_COLS,
)
from src.models.prechecks import (
    check_missing_values,
    descriptive_stats,
    correlation_matrix,
    plot_correlation_heatmap,
    calculate_vif,
    standardize_features,
    run_adf_tests,
    infer_integration_order,
)


def main() -> None:
    tables_dir = PRECHECK_DIR / "tables"
    figures_dir = PRECHECK_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    diff_df = pd.read_csv(MODELLING_DIFF_FILE)
    level_df = pd.read_csv(MODELLING_LEVEL_FILE)

    diff_feature_cols = [
        Y_COL,
        "d_pm25_lag0", "d_pm25_lag1", "d_pm25_lag2",
        "d_ozone_lag0", "d_ozone_lag1", "d_ozone_lag2",
        "d_household_pm_lag0", "d_household_pm_lag1", "d_household_pm_lag2",
        "d_ageing_65_plus_lag0", "d_ageing_65_plus_lag1", "d_ageing_65_plus_lag2",
        "d_gov_health_exp_pct_gdp_lag0", "d_gov_health_exp_pct_gdp_lag1", "d_gov_health_exp_pct_gdp_lag2",
        "trend",
    ]

    missing_df = check_missing_values(diff_df)
    missing_df.to_csv(tables_dir / "prechecks_missing_diff.csv", index=False)

    desc_df = descriptive_stats(diff_df, diff_feature_cols)
    desc_df.to_csv(tables_dir / "prechecks_descriptive_diff.csv", index=False)

    corr_cols = [Y_COL] + [col for col in diff_feature_cols if col != Y_COL]
    corr_df = correlation_matrix(diff_df, corr_cols)
    corr_df.to_csv(tables_dir / "prechecks_correlation_diff.csv")
    plot_correlation_heatmap(corr_df, figures_dir / "prechecks_correlation_diff.png")

    vif_cols = [col for col in diff_feature_cols if col not in [Y_COL]]
    scaled_vif_input = standardize_features(diff_df, vif_cols)
    scaled_vif_input.to_csv(tables_dir / "prechecks_scaled_vif_input.csv", index=False)

    initial_vif_df = calculate_vif(scaled_vif_input, list(scaled_vif_input.columns))
    initial_vif_df.to_csv(tables_dir / "prechecks_full_feature_vif.csv", index=False)

    level_integration_df = infer_integration_order(level_df, ["crd_daly_rate"] + BASE_LEVEL_X_COLS)
    level_integration_df.to_csv(tables_dir / "prechecks_integration_order_screening.csv", index=False)

    diff_adf_df = run_adf_tests(diff_df, [c for c in diff_df.columns if c.startswith("d_")])
    diff_adf_df.to_csv(tables_dir / "prechecks_adf_diff.csv", index=False)

    print("\n=== Initial full-feature VIF ===")
    print(initial_vif_df)
    print("\n=== Integration order screening ===")
    print(level_integration_df)
    print("\n=== ADF results for differenced modelling variables ===")
    print(diff_adf_df)
    print("\nPrechecks completed successfully.")

if __name__ == "__main__":
    main()
