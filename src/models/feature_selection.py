"""Feature screening utilities for nested-validation OLS pipelines."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.prechecks import vif_feature_selection


def _prepare_model_df(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    model_df = df[[y_col] + x_cols].copy()
    for col in [y_col] + x_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    return model_df.dropna().reset_index(drop=True)


def run_elastic_net_grid_search(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    alpha_grid: Iterable[float],
    l1_grid: Iterable[float],
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Run an explicit in-sample grid search (exploratory, not leak-safe for validation)."""
    model_df = _prepare_model_df(df, y_col, x_cols)
    X = model_df[x_cols]
    y = model_df[y_col]

    rows = []
    best = None

    for alpha, l1_ratio in itertools.product(alpha_grid, l1_grid):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=200000, random_state=random_state)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipe.fit(X, y)
        preds = pipe.predict(X)
        coefs = pipe.named_steps["enet"].coef_
        selected = [x_cols[i] for i, coef in enumerate(coefs) if abs(coef) > 1e-8]
        mse = float(np.mean((y - preds) ** 2))

        rows.append({
            "alpha": float(alpha),
            "l1_ratio": float(l1_ratio),
            "n_selected": len(selected),
            "selected_features": " | ".join(selected),
            "screening_mse": mse,
        })

        if len(selected) == 0:
            continue
        if best is None or mse < best["screening_mse"]:
            best = {
                "alpha": float(alpha),
                "l1_ratio": float(l1_ratio),
                "selected_features": selected,
                "screening_mse": mse,
            }

    results_df = pd.DataFrame(rows).sort_values(["screening_mse", "n_selected"], ascending=[True, True]).reset_index(drop=True)

    if best is None:
        best = {
            "alpha": float(list(alpha_grid)[0]),
            "l1_ratio": float(list(l1_grid)[0]),
            "selected_features": [x_cols[0]],
            "screening_mse": np.nan,
        }

    return results_df, best["selected_features"], best


def elastic_net_select_train_only(
    train_df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    alpha_grid: Iterable[float],
    l1_grid: Iterable[float],
    inner_cv_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[str], dict]:
    """Select features using Elastic Net CV on the training fold only."""
    model_df = _prepare_model_df(train_df, y_col, x_cols)
    X = model_df[x_cols]
    y = model_df[y_col]
    if X.empty:
        raise ValueError("Training data are empty after numeric coercion and NA dropping.")

    n_splits = max(2, min(inner_cv_splits, len(model_df)))
    cv = KFold(n_splits=n_splits, shuffle=False)

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "enetcv",
            ElasticNetCV(
                l1_ratio=list(l1_grid),
                alphas=list(alpha_grid),
                cv=cv,
                max_iter=200000,
                random_state=random_state,
            ),
        ),
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)

    fitted = model.named_steps["enetcv"]
    selected = [x_cols[i] for i, coef in enumerate(fitted.coef_) if abs(coef) > 1e-8]
    if not selected:
        selected = [x_cols[int(np.argmax(np.abs(fitted.coef_)))] ] if np.any(np.abs(fitted.coef_) > 0) else [x_cols[0]]

    meta = {
        "alpha": float(fitted.alpha_),
        "l1_ratio": float(fitted.l1_ratio_),
        "n_selected": len(selected),
        "selected_features": selected,
        "inner_cv_splits": n_splits,
    }
    return selected, meta


def backward_elimination(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    pvalue_threshold: float = 0.10,
) -> tuple[pd.DataFrame, list[str]]:
    model_df = _prepare_model_df(df, y_col, x_cols)
    selected = list(x_cols)
    rows = []
    step = 0

    while len(selected) > 1:
        X = sm.add_constant(model_df[selected], has_constant="add")
        model = sm.OLS(model_df[y_col], X).fit()
        pvalues = model.pvalues.drop("const", errors="ignore")
        worst_var = pvalues.idxmax()
        worst_p = float(pvalues.max())

        rows.append({
            "step": step,
            "removed_variable": worst_var if worst_p > pvalue_threshold else "",
            "max_p_value": worst_p,
            "aic": float(model.aic),
            "bic": float(model.bic),
            "remaining_variables": " | ".join(selected),
        })

        if worst_p <= pvalue_threshold:
            break
        selected.remove(worst_var)
        step += 1

    return pd.DataFrame(rows), selected


def random_forest_importance(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    n_estimators: int = 500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    model_df = _prepare_model_df(df, y_col, x_cols)
    X = model_df[x_cols]
    y = model_df[y_col]

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)

    imp_df = pd.DataFrame({
        "variable": x_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    median_importance = imp_df["importance"].median()
    selected = imp_df.loc[imp_df["importance"] >= median_importance, "variable"].tolist()
    if len(selected) == 0:
        selected = [imp_df.iloc[0]["variable"]]

    return imp_df, selected


def vif_screening_path(
    df: pd.DataFrame,
    x_cols: list[str],
    threshold: float = 10.0,
    scale_first: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    _, vif_path_df, selected = vif_feature_selection(
        df=df,
        columns=x_cols,
        threshold=threshold,
        scale_first=scale_first,
    )
    return vif_path_df, selected
