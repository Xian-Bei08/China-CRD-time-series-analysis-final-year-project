"""Validation helpers for nested-LOOCV OLS pipelines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.models.feature_selection import (
    backward_elimination,
    elastic_net_select_train_only,
    random_forest_importance,
    vif_screening_path,
)


def prepare_validation_data(df: pd.DataFrame, y_col: str, x_cols: list[str], year_col: str = "year") -> pd.DataFrame:
    required_cols = [year_col, y_col] + x_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for validation: {missing_cols}")

    model_df = df[required_cols].copy().sort_values(year_col).reset_index(drop=True)
    for col in [y_col] + x_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    return model_df.dropna(subset=[y_col] + x_cols).reset_index(drop=True)

def _select_features_for_path(
    train_df: pd.DataFrame,
    y_col: str,
    candidate_cols: list[str],
    path_name: str,
    *,
    alpha_grid: list[float],
    l1_grid: list[float],
    inner_cv_splits: int,
    random_state: int,
    backward_pvalue_threshold: float,
    random_forest_n_estimators: int,
    vif_threshold: float,
) -> tuple[list[str], dict]:
    if path_name == "elastic_net":
        selected, meta = elastic_net_select_train_only(
            train_df=train_df,
            y_col=y_col,
            x_cols=candidate_cols,
            alpha_grid=alpha_grid,
            l1_grid=l1_grid,
            inner_cv_splits=inner_cv_splits,
            random_state=random_state,
        )
        return selected, meta
    if path_name == "backward_elimination":
        path_df, selected = backward_elimination(
            df=train_df,
            y_col=y_col,
            x_cols=candidate_cols,
            pvalue_threshold=backward_pvalue_threshold,
        )
        return selected, {
            "n_steps": int(len(path_df)),
            "final_bic": float(path_df.iloc[-1]["bic"]) if not path_df.empty else np.nan,
        }
    if path_name == "random_forest":
        imp_df, selected = random_forest_importance(
            df=train_df,
            y_col=y_col,
            x_cols=candidate_cols,
            n_estimators=random_forest_n_estimators,
            random_state=random_state,
        )
        return selected, {
            "median_importance": float(imp_df["importance"].median()),
            "top_variable": str(imp_df.iloc[0]["variable"]),
        }
    if path_name == "vif_filtering":
        path_df, selected = vif_screening_path(
            df=train_df,
            x_cols=candidate_cols,
            threshold=vif_threshold,
            scale_first=True,
        )
        return selected, {
            "n_steps": int(len(path_df)),
            "final_max_vif": float(path_df.iloc[-1]["vif"]) if not path_df.empty and "vif" in path_df.columns else np.nan,
        }
    raise ValueError(f"Unknown path name: {path_name}")


def nested_loocv_single_path(
    df: pd.DataFrame,
    y_col: str,
    candidate_cols: list[str],
    path_name: str,
    year_col: str = "year",
    *,
    alpha_grid: list[float],
    l1_grid: list[float],
    inner_cv_splits: int,
    random_state: int,
    backward_pvalue_threshold: float,
    random_forest_n_estimators: int,
    vif_threshold: float,
) -> pd.DataFrame:
    model_df = prepare_validation_data(df, y_col=y_col, x_cols=candidate_cols, year_col=year_col)
    rows = []

    for i in range(len(model_df)):
        train_df = model_df.drop(index=i).reset_index(drop=True)
        test_row = model_df.iloc[[i]].reset_index(drop=True)

        selected_cols, meta = _select_features_for_path(
            train_df=train_df,
            y_col=y_col,
            candidate_cols=candidate_cols,
            path_name=path_name,
            alpha_grid=alpha_grid,
            l1_grid=l1_grid,
            inner_cv_splits=inner_cv_splits,
            random_state=random_state,
            backward_pvalue_threshold=backward_pvalue_threshold,
            random_forest_n_estimators=random_forest_n_estimators,
            vif_threshold=vif_threshold,
        )

        train_clean = prepare_validation_data(train_df, y_col=y_col, x_cols=selected_cols, year_col=year_col)
        test_clean = prepare_validation_data(test_row, y_col=y_col, x_cols=selected_cols, year_col=year_col)
        X_train = sm.add_constant(train_clean[selected_cols], has_constant="add")
        y_train = train_clean[y_col]
        X_test = sm.add_constant(test_clean[selected_cols], has_constant="add")

        model = sm.OLS(y_train, X_train).fit()
        pred = float(model.predict(X_test).iloc[0])
        actual = float(test_clean[y_col].iloc[0])

        row = {
            "path": path_name,
            "year": int(test_clean[year_col].iloc[0]),
            "train_size": len(train_clean),
            "n_selected": len(selected_cols),
            "selected_variables": " | ".join(selected_cols),
            "actual": actual,
            "predicted": pred,
            "error": actual - pred,
            "abs_error": abs(actual - pred),
            "squared_error": (actual - pred) ** 2,
            "ape": abs((actual - pred) / actual) * 100 if actual != 0 else np.nan,
        }
        for key, value in meta.items():
            if isinstance(value, list):
                row[f"meta_{key}"] = " | ".join(map(str, value))
            else:
                row[f"meta_{key}"] = value
        rows.append(row)

    return pd.DataFrame(rows)


def metrics_table(validation_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    me = validation_df["error"].mean()
    mae = validation_df["abs_error"].mean()
    rmse = float(np.sqrt(validation_df["squared_error"].mean()))
    mape = validation_df["ape"].mean()
    directional_accuracy = float((np.sign(validation_df["actual"]) == np.sign(validation_df["predicted"])).mean() * 100)

    return pd.DataFrame({
        "model": [model_name],
        "n_forecasts": [len(validation_df)],
        "mean_error": [me],
        "mae": [mae],
        "rmse": [rmse],
        "mape_percent": [mape],
        "directional_accuracy_percent": [directional_accuracy],
    })


def selection_frequency_table(detail_df: pd.DataFrame, path_name: str) -> pd.DataFrame:
    counts: dict[str, int] = {}
    total_folds = len(detail_df)
    for value in detail_df["selected_variables"].fillna(""):
        vars_for_fold = [item.strip() for item in str(value).split("|") if item.strip()]
        for var in vars_for_fold:
            counts[var] = counts.get(var, 0) + 1
    out = pd.DataFrame(
        [{"path": path_name, "variable": key, "selected_in_folds": val, "selection_rate": val / total_folds} for key, val in counts.items()]
    )
    if out.empty:
        return pd.DataFrame(columns=["path", "variable", "selected_in_folds", "selection_rate"])
    return out.sort_values(["selected_in_folds", "variable"], ascending=[False, True]).reset_index(drop=True)

def _score_inner_cv_for_path(
    train_outer_df: pd.DataFrame,
    y_col: str,
    candidate_cols: list[str],
    path_name: str,
    year_col: str = "year",
    *,
    alpha_grid: list[float],
    l1_grid: list[float],
    inner_cv_splits: int,
    random_state: int,
    backward_pvalue_threshold: float,
    random_forest_n_estimators: int,
    vif_threshold: float,
    metric: str = "mae",
) -> tuple[float, pd.DataFrame]:
    """Run an inner LOOCV on the outer-training fold for one path and return its score."""
    inner_detail_df = nested_loocv_single_path(
        df=train_outer_df,
        y_col=y_col,
        candidate_cols=candidate_cols,
        path_name=path_name,
        year_col=year_col,
        alpha_grid=alpha_grid,
        l1_grid=l1_grid,
        inner_cv_splits=inner_cv_splits,
        random_state=random_state,
        backward_pvalue_threshold=backward_pvalue_threshold,
        random_forest_n_estimators=random_forest_n_estimators,
        vif_threshold=vif_threshold,
    )
    metric = metric.lower()
    if metric == "mae":
        score = float(inner_detail_df["abs_error"].mean())
    elif metric == "rmse":
        score = float(np.sqrt(inner_detail_df["squared_error"].mean()))
    elif metric == "mape":
        score = float(inner_detail_df["ape"].mean())
    else:
        raise ValueError(f"Unsupported inner-selection metric: {metric}")
    return score, inner_detail_df


def nested_loocv_select_best_path(
    df: pd.DataFrame,
    y_col: str,
    candidate_cols: list[str],
    path_names: list[str],
    year_col: str = "year",
    *,
    alpha_grid: list[float],
    l1_grid: list[float],
    inner_cv_splits: int,
    random_state: int,
    backward_pvalue_threshold: float,
    random_forest_n_estimators: int,
    vif_threshold: float,
    selection_metric: str = "mae",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """True nested LOOCV with inner-loop path selection."""
    model_df = prepare_validation_data(df, y_col=y_col, x_cols=candidate_cols, year_col=year_col)
    outer_rows = []
    inner_rows = []

    for i in range(len(model_df)):
        train_outer_df = model_df.drop(index=i).reset_index(drop=True)
        test_outer_df = model_df.iloc[[i]].reset_index(drop=True)

        path_scores = []
        for path_name in path_names:
            score, inner_detail_df = _score_inner_cv_for_path(
                train_outer_df=train_outer_df,
                y_col=y_col,
                candidate_cols=candidate_cols,
                path_name=path_name,
                year_col=year_col,
                alpha_grid=alpha_grid,
                l1_grid=l1_grid,
                inner_cv_splits=inner_cv_splits,
                random_state=random_state,
                backward_pvalue_threshold=backward_pvalue_threshold,
                random_forest_n_estimators=random_forest_n_estimators,
                vif_threshold=vif_threshold,
                metric=selection_metric,
            )
            path_scores.append({"path": path_name, "score": score})
            inner_rows.append({
                "outer_test_year": int(test_outer_df[year_col].iloc[0]),
                "candidate_path": path_name,
                "inner_selection_metric": selection_metric.lower(),
                "inner_cv_score": score,
                "inner_n_forecasts": len(inner_detail_df),
                "inner_mean_n_selected": float(inner_detail_df["n_selected"].mean()),
            })

        score_df = pd.DataFrame(path_scores).sort_values(["score", "path"]).reset_index(drop=True)
        best_path = str(score_df.iloc[0]["path"])

        selected_cols, meta = _select_features_for_path(
            train_df=train_outer_df,
            y_col=y_col,
            candidate_cols=candidate_cols,
            path_name=best_path,
            alpha_grid=alpha_grid,
            l1_grid=l1_grid,
            inner_cv_splits=inner_cv_splits,
            random_state=random_state,
            backward_pvalue_threshold=backward_pvalue_threshold,
            random_forest_n_estimators=random_forest_n_estimators,
            vif_threshold=vif_threshold,
        )

        train_clean = prepare_validation_data(train_outer_df, y_col=y_col, x_cols=selected_cols, year_col=year_col)
        test_clean = prepare_validation_data(test_outer_df, y_col=y_col, x_cols=selected_cols, year_col=year_col)
        X_train = sm.add_constant(train_clean[selected_cols], has_constant="add")
        y_train = train_clean[y_col]
        X_test = sm.add_constant(test_clean[selected_cols], has_constant="add")

        model = sm.OLS(y_train, X_train).fit()
        pred = float(model.predict(X_test).iloc[0])
        actual = float(test_clean[y_col].iloc[0])

        row = {
            "outer_selected_path": best_path,
            "year": int(test_clean[year_col].iloc[0]),
            "train_size": len(train_clean),
            "n_selected": len(selected_cols),
            "selected_variables": " | ".join(selected_cols),
            "actual": actual,
            "predicted": pred,
            "error": actual - pred,
            "abs_error": abs(actual - pred),
            "squared_error": (actual - pred) ** 2,
            "ape": abs((actual - pred) / actual) * 100 if actual != 0 else np.nan,
            "selection_metric": selection_metric.lower(),
            "best_inner_cv_score": float(score_df.iloc[0]["score"]),
        }
        for key, value in meta.items():
            row[f"meta_{key}"] = " | ".join(map(str, value)) if isinstance(value, list) else value
        for rank, score_row in enumerate(score_df.itertuples(index=False), start=1):
            row[f"candidate_{rank}_path"] = score_row.path
            row[f"candidate_{rank}_score"] = float(score_row.score)
        outer_rows.append(row)

    return pd.DataFrame(outer_rows), pd.DataFrame(inner_rows)
