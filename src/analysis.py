"""
analysis.py
Statistical analysis, modeling, evaluation metrics, and visualization.
"""

import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.inspection import permutation_importance

ACCENT = "#c0392b"
MUTED = "#9a9a9a"


def configure_plot_style():
    """Configures clean, publication-ready Matplotlib aesthetics."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e5e5e5",
        "axes.axisbelow": True,
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    })


def evaluate_metrics(name: str, y_true, pred, proba=None) -> Dict[str, Any]:
    """Calculates operational and statistical classification metrics."""
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    
    return {
        "model": name,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "accuracy": float((tp + tn) / len(y_true)),
        "als_dispatch_rate_pct": float(pred.mean() * 100),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "pr_auc": float(average_precision_score(y_true, proba)) if proba is not None else np.nan,
    }


def threshold_for_recall(y_true, proba, target: float = 0.85) -> float:
    """Finds the maximum threshold (minimizing ALS dispatches) achieving recall >= target."""
    _, rec, thr = precision_recall_curve(y_true, proba)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) > 0 else float(thr[0])


def build_models(preprocessor_fn, seed: int = 42) -> Dict[str, Any]:
    """Defines classifier pipelines with balanced class weighting."""
    return {
        "Logistic Regression": lambda: Pipeline([
            ("prep", preprocessor_fn()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
        ]),
        "Random Forest": lambda: Pipeline([
            ("prep", preprocessor_fn()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )),
        ]),
    }


def rule_baseline(frame: pd.DataFrame, high_risk_vehicles: List[str]) -> np.ndarray:
    """Baseline 2 heuristic: Send ALS if vehicle is high-risk or incident occurred at night."""
    return (frame["vehicle_type"].isin(high_risk_vehicles) | (frame["is_night"] == 1)).astype(int).values


def run_full_evaluation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: List[str],
    target_col: str,
    models_dict: Dict[str, Any],
    target_recall: float = 0.85,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict[str, float], Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Executes validation tuning on 2021, selects best model by PR-AUC,
    then evaluates on held-out 2022 test set.
    """
    # 1. Validation tuning
    fit_part = train[train["year"] <= 2020]
    val_part = train[train["year"] == 2021]
    
    val_rows, thresholds = {}, {}
    for name, make_fn in models_dict.items():
        m = make_fn().fit(fit_part[features], fit_part[target_col])
        p = m.predict_proba(val_part[features])[:, 1]
        thr = threshold_for_recall(val_part[target_col], p, target_recall)
        thresholds[name] = thr
        row = evaluate_metrics(name, val_part[target_col], (p >= thr).astype(int), p)
        row["threshold"] = thr
        val_rows[name] = row

    val_df = pd.DataFrame(list(val_rows.values())).set_index("model")
    best_model_name = str(val_df["pr_auc"].idxmax())

    # High-risk vehicle types definition for baseline
    rates_vt = train.groupby("vehicle_type")[target_col].mean()
    high_risk_vehicles = sorted(rates_vt[rates_vt > train[target_col].mean()].index)

    # 2. Retrain on full train (2019-2021) and evaluate on test (2022)
    fitted_models, test_probas = {}, {}
    for name, make_fn in models_dict.items():
        fitted = make_fn().fit(train[features], train[target_col])
        fitted_models[name] = fitted
        test_probas[name] = fitted.predict_proba(test[features])[:, 1]

    y_test = test[target_col].values
    results = [
        evaluate_metrics("Baseline 1: Always Non-fatal", y_test, np.zeros(len(test), dtype=int)),
        evaluate_metrics("Baseline 2: Simple Rule (Risky Vehicle / Night)", y_test, rule_baseline(test, high_risk_vehicles)),
    ]
    for name in models_dict:
        pred = (test_probas[name] >= thresholds[name]).astype(int)
        results.append(evaluate_metrics(name, y_test, pred, test_probas[name]))

    test_df = pd.DataFrame(results).set_index("model")
    return val_df, test_df, best_model_name, thresholds, fitted_models, test_probas


def plot_baseline_vs_models(
    results_table: pd.DataFrame,
    best_model: str,
    target_recall: float,
    output_path: str,
):
    """Generates the R4 baseline comparison horizontal bar chart."""
    configure_plot_style()
    tbl = results_table.iloc[::-1]
    colors = [ACCENT if n == best_model else ("#e8a59d" if "Baseline" not in n else "#bdbdbd") for n in tbl.index]

    labels = tbl.index.tolist()
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.barh(labels, tbl["recall"], color=colors, height=0.6)
    
    for i, (r, a) in enumerate(zip(tbl["recall"], tbl["als_dispatch_rate_pct"])):
        ax.text(r + 0.015, i, f"Recall: {r:.2f}  |  Dispatches ALS to {a:.0f}% cases", va="center", fontsize=9, color="#333")
        
    ax.axvline(target_recall, color=MUTED, ls="--", lw=1.5)
    ax.set_xlim(0, 1.4)
    ax.set_xlabel(f"Recall of fatal accidents (Test 2022)   ·   Dashed line = Target ({target_recall:.2f})")
    ax.set_title("Baseline vs Models: Share of Fatal Accidents Caught vs Dispatch Workload", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tradeoff_curve(
    y_test: np.ndarray,
    best_proba: np.ndarray,
    chosen_stats: pd.Series,
    rule_stats: pd.Series,
    output_path: str,
):
    """Generates the ALS workload vs fatal cases caught trade-off curve."""
    configure_plot_style()
    order = np.argsort(-best_proba)
    caught_cum = np.cumsum(y_test[order]) / y_test.sum()
    alert_share = np.linspace(0, 1, len(y_test))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alert_share * 100, caught_cum * 100, color=ACCENT, lw=2.2, label="Model Trade-off Curve")
    ax.plot([0, 100], [0, 100], color=MUTED, ls="--", lw=1.5, label="Random Dispatch")
    
    ax.scatter([chosen_stats["als_dispatch_rate_pct"]], [chosen_stats["recall"] * 100], s=80, color=ACCENT, edgecolor="white", lw=2, zorder=3)
    ax.annotate(
        f"Chosen Threshold\nALS {chosen_stats['als_dispatch_rate_pct']:.0f}% → Caught {chosen_stats['recall']*100:.0f}%",
        (chosen_stats["als_dispatch_rate_pct"], chosen_stats["recall"] * 100),
        xytext=(15, -30), textcoords="offset points", fontsize=9, color="#222"
    )
    
    ax.scatter([rule_stats["als_dispatch_rate_pct"]], [rule_stats["recall"] * 100], s=70, color="#555", edgecolor="white", lw=2, zorder=3)
    ax.annotate(
        f"Simple Rule\nALS {rule_stats['als_dispatch_rate_pct']:.0f}% → Caught {rule_stats['recall']*100:.0f}%",
        (rule_stats["als_dispatch_rate_pct"], rule_stats["recall"] * 100),
        xytext=(15, 5), textcoords="offset points", fontsize=9, color="#555"
    )
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Accidents sent an ALS team (%)   [Dispatch Workload]")
    ax.set_ylabel("Fatal accidents caught (%)   [Recall]")
    ax.set_title("Operational Trade-off: ALS Workload vs Fatal Cases Caught (Test 2022)", loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
