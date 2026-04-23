"""Compare three feature-screening paths: Elastic Net, backward elimination, and random forest importance."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    MODELLING_DIFF_FILE,
    FEATURE_SCREENING_DIR,
    Y_COL,
    ELASTIC_NET_ALPHA_GRID,
    ELASTIC_NET_L1_GRID,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_STATE,
    BACKWARD_PVALUE_THRESHOLD,
    VIF_THRESHOLD,
)
from src.models.feature_selection import (
    run_elastic_net_grid_search,
    backward_elimination,
    random_forest_importance,
    vif_screening_path,
)


def main() -> None:
    tables_dir = FEATURE_SCREENING_DIR / "tables"
    figures_dir = FEATURE_SCREENING_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MODELLING_DIFF_FILE)
    x_cols = [
        "d_pm25_lag0",
        "d_pm25_lag1",
        "d_pm25_lag2",

        "d_gov_health_exp_pct_gdp_lag0",
        "d_gov_health_exp_pct_gdp_lag1",
        "d_gov_health_exp_pct_gdp_lag2",

        "d_household_pm_lag1",
        "d_household_pm_lag2",

        "d_ageing_65_plus_lag1",

        "trend",
    ]

    elastic_results_df, elastic_selected, elastic_meta = run_elastic_net_grid_search(
        df=df,
        y_col=Y_COL,
        x_cols=x_cols,
        alpha_grid=ELASTIC_NET_ALPHA_GRID,
        l1_grid=ELASTIC_NET_L1_GRID,
        random_state=RANDOM_STATE,
    )
    elastic_results_df.to_csv(tables_dir / "elastic_net_tuning_results.csv", index=False)

    backward_path_df, backward_selected = backward_elimination(
        df=df,
        y_col=Y_COL,
        x_cols=x_cols,
        pvalue_threshold=BACKWARD_PVALUE_THRESHOLD,
    )
    backward_path_df.to_csv(tables_dir / "backward_selection_path.csv", index=False)

    rf_importance_df, rf_selected = random_forest_importance(
        df=df,
        y_col=Y_COL,
        x_cols=x_cols,
        n_estimators=RANDOM_FOREST_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    rf_importance_df.to_csv(tables_dir / "rf_feature_importance.csv", index=False)

    vif_path_df, vif_selected = vif_screening_path(
        df=df,
        x_cols=x_cols,
        threshold=VIF_THRESHOLD,
        scale_first=True,
    )
    vif_path_df.to_csv(tables_dir / "vif_selection_path.csv", index=False)
    pd.DataFrame({"selected_variable": vif_selected}).to_csv(
        tables_dir / "vif_selected_variables.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(rf_importance_df["variable"], rf_importance_df["importance"])
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Importance")
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.savefig(figures_dir / "rf_importance_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    summary_df = pd.DataFrame([
        {
            "path": "elastic_net",
            "selected_variables": " | ".join(elastic_selected),
            "n_selected": len(elastic_selected),
            "screening_metric": elastic_meta["screening_mse"],
            "notes": f"best alpha={elastic_meta['alpha']}, best l1_ratio={elastic_meta['l1_ratio']}",
        },
        {
            "path": "backward_elimination",
            "selected_variables": " | ".join(backward_selected),
            "n_selected": len(backward_selected),
            "screening_metric": float(backward_path_df.iloc[-1]["bic"]) if not backward_path_df.empty else None,
            "notes": f"p-threshold={BACKWARD_PVALUE_THRESHOLD}",
        },
        {
            "path": "random_forest",
            "selected_variables": " | ".join(rf_selected),
            "n_selected": len(rf_selected),
            "screening_metric": float(rf_importance_df["importance"].sum()),
            "notes": "selected features with above-median importance",
        },
        {
            "path": "vif_filtering",
            "selected_variables": " | ".join(vif_selected),
            "n_selected": len(vif_selected),
            "screening_metric": float(vif_path_df.iloc[-1]["max_vif"]) if not vif_path_df.empty and "max_vif" in vif_path_df.columns else None,
            "notes": f"iterative VIF filtering with threshold={VIF_THRESHOLD}",
        },
    ])
    summary_df.to_csv(tables_dir / "feature_selection_comparison_summary.csv", index=False)

    print("\n=== Feature screening summary ===")
    print(summary_df)


if __name__ == "__main__":
    main()
