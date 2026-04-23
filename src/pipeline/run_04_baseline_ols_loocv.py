"""Run true nested LOOCV for feature-selection paths, then refit the best OLS path on the full sample."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from src.config import (
    MODELLING_DIFF_FILE,
    OLS_LOOCV_DIR,
    Y_COL,
    SCREENING_CANDIDATE_X_COLS,
    ELASTIC_NET_ALPHA_GRID,
    ELASTIC_NET_L1_GRID,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_STATE,
    BACKWARD_PVALUE_THRESHOLD,
    VIF_THRESHOLD,
    INNER_CV_SPLITS,
)
from src.models.feature_selection import (
    backward_elimination,
    elastic_net_select_train_only,
    random_forest_importance,
    vif_screening_path,
)
from src.models.validation import (
    compare_nested_loocv_paths,
    nested_loocv_select_best_path,
    metrics_table,
    selection_frequency_table,
)

PATH_NAMES = ["elastic_net", "backward_elimination", "random_forest", "vif_filtering"]


def _final_full_sample_selection(df: pd.DataFrame, path_name: str) -> tuple[list[str], dict]:
    if path_name == "elastic_net":
        return elastic_net_select_train_only(
            train_df=df,
            y_col=Y_COL,
            x_cols=SCREENING_CANDIDATE_X_COLS,
            alpha_grid=ELASTIC_NET_ALPHA_GRID,
            l1_grid=ELASTIC_NET_L1_GRID,
            inner_cv_splits=INNER_CV_SPLITS,
            random_state=RANDOM_STATE,
        )
    if path_name == "backward_elimination":
        path_df, selected = backward_elimination(
            df=df,
            y_col=Y_COL,
            x_cols=SCREENING_CANDIDATE_X_COLS,
            pvalue_threshold=BACKWARD_PVALUE_THRESHOLD,
        )
        return selected, {"n_steps": len(path_df)}
    if path_name == "random_forest":
        imp_df, selected = random_forest_importance(
            df=df,
            y_col=Y_COL,
            x_cols=SCREENING_CANDIDATE_X_COLS,
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            random_state=RANDOM_STATE,
        )
        top_variable = str(imp_df.iloc[0]["variable"]) if not imp_df.empty else None
        return selected, {"top_variable": top_variable}
    if path_name == "vif_filtering":
        path_df, selected = vif_screening_path(
            df=df,
            x_cols=SCREENING_CANDIDATE_X_COLS,
            threshold=VIF_THRESHOLD,
            scale_first=True,
        )
        return selected, {"n_steps": len(path_df)}
    raise ValueError(f"Unknown path name: {path_name}")


def _save_nested_plots(detail_df: pd.DataFrame, figures_dir) -> None:
    """Save only the final nested-LOOCV plots."""
    plot_df = detail_df.sort_values("year").copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df["year"], plot_df["actual"], marker="o", label="Actual")
    ax.plot(plot_df["year"], plot_df["predicted"], marker="o", label="Predicted")
    ax.set_title("Nested LOOCV: Actual vs Predicted")
    ax.set_xlabel("Year")
    ax.set_ylabel(Y_COL)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "nested_loocv_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df["year"], plot_df["error"], marker="o")
    ax.axhline(0, linestyle="--")
    ax.set_title("Nested LOOCV: Prediction Error Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Prediction error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "nested_loocv_prediction_error_trend.png", dpi=300, bbox_inches="tight")
    plt.close()

def main() -> None:
    tables_dir = OLS_LOOCV_DIR / "tables"
    figures_dir = OLS_LOOCV_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MODELLING_DIFF_FILE)

    outer_detail_df, inner_scores_df = nested_loocv_select_best_path(
        df=df,
        y_col=Y_COL,
        candidate_cols=SCREENING_CANDIDATE_X_COLS,
        path_names=PATH_NAMES,
        year_col="year",
        alpha_grid=ELASTIC_NET_ALPHA_GRID,
        l1_grid=ELASTIC_NET_L1_GRID,
        inner_cv_splits=INNER_CV_SPLITS,
        random_state=RANDOM_STATE,
        backward_pvalue_threshold=BACKWARD_PVALUE_THRESHOLD,
        random_forest_n_estimators=RANDOM_FOREST_N_ESTIMATORS,
        vif_threshold=VIF_THRESHOLD,
        selection_metric="mae",
    )
    outer_detail_df.to_csv(tables_dir / "nested_loocv_prediction_paths.csv", index=False)
    inner_scores_df.to_csv(tables_dir / "nested_loocv_inner_model_selection_scores.csv", index=False)

    nested_metrics_df = metrics_table(outer_detail_df, model_name="nested_best_path")
    nested_metrics_df["selection_metric"] = "mae"
    nested_metrics_df["mean_n_selected"] = float(outer_detail_df["n_selected"].mean())
    nested_metrics_df["most_frequent_outer_path"] = str(outer_detail_df["outer_selected_path"].mode().iloc[0])
    nested_metrics_df.to_csv(tables_dir / "nested_loocv_metrics_summary.csv", index=False)

    nested_freq_df = selection_frequency_table(
        outer_detail_df.rename(columns={"outer_selected_path": "path"}),
        path_name="nested_best_path",
    )
    nested_freq_df.to_csv(tables_dir / "nested_loocv_selection_frequency.csv", index=False)

    final_best_path = str(outer_detail_df["outer_selected_path"].mode().iloc[0])
    best_x_cols, best_meta = _final_full_sample_selection(df, final_best_path)

    pd.DataFrame([{
        "best_path": final_best_path,
        "final_selected_variables": " | ".join(best_x_cols),
        "n_selected": len(best_x_cols),
        **{
            f"meta_{k}": (" | ".join(map(str, v)) if isinstance(v, list) else v)
            for k, v in best_meta.items()
        },
    }]).to_csv(tables_dir / "best_path_final_selection.csv", index=False)

    model_df = df[["year", Y_COL] + best_x_cols].copy().dropna().reset_index(drop=True)
    X = sm.add_constant(model_df[best_x_cols], has_constant="add")
    y = model_df[Y_COL]
    ols_res = sm.OLS(y, X).fit()

    coef_df = pd.DataFrame({
        "variable": ols_res.params.index,
        "coefficient": ols_res.params.values,
        "std_error": ols_res.bse.values,
        "t_value": ols_res.tvalues.values,
        "p_value": ols_res.pvalues.values,
    })
    coef_df.to_csv(tables_dir / f"ols_{final_best_path}_coefficients.csv", index=False)

    with open(tables_dir / f"ols_{final_best_path}_summary.txt", "w", encoding="utf-8") as f:
        f.write(ols_res.summary().as_text())

    _save_nested_plots(outer_detail_df, figures_dir)

    print("\n=== True nested LOOCV summary ===")
    print(nested_metrics_df)
    print("\n=== Final full-sample selection for modal outer best path ===")
    print({"best_path": final_best_path, "final_selected_variables": best_x_cols})


if __name__ == "__main__":
    main()
