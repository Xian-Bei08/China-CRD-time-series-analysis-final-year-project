"""Helpers for reading nested-LOOCV selection outputs."""

from __future__ import annotations

import pandas as pd

from src.config import OLS_LOOCV_DIR, TREND_COL


def load_metrics_summary() -> pd.DataFrame:
    return pd.read_csv(OLS_LOOCV_DIR / 'tables' / 'nested_loocv_metrics_summary.csv')


def load_best_path() -> str:
    metrics_df = load_metrics_summary()
    return str(metrics_df.iloc[0]['model'])


def load_final_selected_variables() -> list[str]:
    selection_df = pd.read_csv(OLS_LOOCV_DIR / 'tables' / 'best_path_final_selection.csv')
    raw = str(selection_df.iloc[0]['final_selected_variables'])
    return [item.strip() for item in raw.split('|') if item.strip()]


def map_diff_to_level_variables(selected_variables: list[str]) -> list[str]:
    level_variables: list[str] = []
    for variable in selected_variables:
        if variable == TREND_COL:
            if TREND_COL not in level_variables:
                level_variables.append(TREND_COL)
            continue
        if variable.startswith('d_') and '_lag' in variable:
            base = variable[2:].split('_lag')[0]
            if base not in level_variables:
                level_variables.append(base)
    if TREND_COL not in level_variables:
        level_variables.append(TREND_COL)
    return level_variables
