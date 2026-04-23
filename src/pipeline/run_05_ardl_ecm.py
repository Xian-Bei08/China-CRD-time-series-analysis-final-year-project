"""Fit ARDL-ECM using the best nested-LOOCV path and its final full-sample variable set."""

from __future__ import annotations

import pandas as pd

from src.config import MODELLING_LEVEL_FILE, BOUNDS_ECM_DIR, LEVEL_Y_COL
from src.models.bounds_ecm import (
    fit_ardl_then_bounds,
    bounds_result_table,
    uecm_coefficients_table,
    ecm_speed_of_adjustment_table,
    fitted_differences_export,
    plot_uecm_actual_vs_fitted,
    save_text,
    bounds_interpretation_text,
    ci_table,
)
from src.models.selection_results import (
    load_best_path,
    load_final_selected_variables,
    map_diff_to_level_variables,
)


def main() -> None:
    tables_dir = BOUNDS_ECM_DIR / 'tables'
    figures_dir = BOUNDS_ECM_DIR / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    level_df = pd.read_csv(MODELLING_LEVEL_FILE)
    best_path = load_best_path()
    selected_variables = load_final_selected_variables()
    ardl_x_cols = map_diff_to_level_variables(selected_variables)

    results = fit_ardl_then_bounds(
        df=level_df,
        y_col=LEVEL_Y_COL,
        x_cols=ardl_x_cols,
        maxlag=2,
        maxorder=2,
        ic='bic',
        trend='c',
        year_col='year',
        causal=False,
        bounds_case=3,
    )

    model_df = results['model_df']
    selector = results['selector']
    ardl_res = results['ardl_res']
    uecm_res = results['uecm_res']
    bounds_res = results['bounds_res']

    bounds_df = bounds_result_table(bounds_res)
    bounds_df.to_csv(tables_dir / 'bounds_test_results.csv', index=False)
    bounds_res.crit_vals.to_csv(tables_dir / 'bounds_test_critical_values.csv')
    uecm_coefficients_table(uecm_res).to_csv(tables_dir / 'uecm_coefficients.csv', index=False)
    ecm_speed_of_adjustment_table(uecm_res).to_csv(tables_dir / 'ecm_speed_of_adjustment.csv', index=False)

    fitted_df = fitted_differences_export(
        model_df=model_df,
        uecm_res=uecm_res,
        y_col=LEVEL_Y_COL,
        year_col='year',
    )
    fitted_df.to_csv(tables_dir / 'uecm_actual_vs_fitted.csv', index=False)

    save_text(ardl_res.summary().as_text(), tables_dir / 'ardl_summary.txt')
    save_text(uecm_res.summary().as_text(), tables_dir / 'uecm_summary.txt')
    save_text(str(bounds_res), tables_dir / 'bounds_test.txt')
    save_text(bounds_interpretation_text(bounds_res, case=3), tables_dir / 'bounds_interpretation.txt')
    save_text(str(selector.model.ardl_order), tables_dir / 'ardl_selected_order.txt')

    try:
        ci_df = ci_table(uecm_res)
        ci_df.to_csv(tables_dir / 'cointegrating_vector.csv', index=False)
        save_text(str(uecm_res.ci_summary()), tables_dir / 'cointegrating_vector_summary.txt')
    except Exception as exc:
        save_text(f'Cointegrating vector summary not available.\nReason: {exc}\n', tables_dir / 'cointegrating_vector_summary.txt')

    plot_uecm_actual_vs_fitted(
        fitted_df=fitted_df,
        y_col=LEVEL_Y_COL,
        output_file=figures_dir / 'uecm_actual_vs_fitted.png',
        year_col='year',
    )

    print(f'Best LOOCV path: {best_path}')
    print(f'Selected short-run variables: {selected_variables}')
    print(f'Level variables sent to ARDL-ECM: {ardl_x_cols}')
    print(bounds_df)


if __name__ == '__main__':
    main()
