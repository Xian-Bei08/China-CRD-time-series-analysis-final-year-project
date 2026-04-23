"""Produce dissertation diagnostics and visuals, including the required CUSUM stability plot."""

from __future__ import annotations

import pandas as pd

from src.config import MODELLING_DIFF_FILE, DIAGNOSTICS_DIR, Y_COL
from src.models.diagnostics import fit_ols_for_diagnostics, cusum_dataframe, plot_cusum
from src.models.selection_results import load_best_path, load_final_selected_variables


def main() -> None:
    tables_dir = DIAGNOSTICS_DIR / 'tables'
    figures_dir = DIAGNOSTICS_DIR / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MODELLING_DIFF_FILE)
    best_path = load_best_path()
    selected_variables = load_final_selected_variables()

    ols_res, _ = fit_ols_for_diagnostics(df=df, y_col=Y_COL, x_cols=selected_variables)
    cusum_df = cusum_dataframe(ols_res)
    cusum_df.to_csv(tables_dir / 'cusum_stability_series.csv', index=False)
    plot_cusum(cusum_df, figures_dir / 'cusum_stability_plot.png')

    print(f'Generated CUSUM plot for best path: {best_path}')
    print(f'Variables used for diagnostics: {selected_variables}')


if __name__ == '__main__':
    main()
