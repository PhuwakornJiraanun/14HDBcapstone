"""
main.py
Main pipeline runner for Thailand Road Accident Fatality Prediction (1669 Dispatch).
SDG 11 - Sustainable Cities and Communities
"""

import os
import argparse
import json
import pandas as pd
from preprocess import (
    load_dataset,
    clean_and_engineer_features,
    temporal_split,
    group_rare_categories,
    build_pipeline_preprocessor,
    TARGET,
)
from analysis import (
    build_models,
    run_full_evaluation,
    plot_baseline_vs_models,
    plot_tradeoff_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Road Accident Fatality Prediction Pipeline (SDG 11 / 1669 Triage)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to raw CSV dataset (default: checks data/raw or kagglehub)",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.85,
        help="Target minimum recall for fatal accidents (default: 0.85)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("results", "output"),
        help="Directory to save evaluation results and metrics",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=os.path.join("results", "figures"),
        help="Directory to save generated charts and figures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    print("=" * 70)
    print(" 14HDBcapstone: Fatal Road Accident Prediction for 1669 Dispatch")
    print(" UN SDG 11: Sustainable Cities and Communities (Target 11.2)")
    print("=" * 70)

    # 1. Load Data
    raw_df = load_dataset(args.data_path)
    print(f"Total raw records loaded: {len(raw_df):,}")

    # 2. Preprocess & Feature Engineering
    df, cat_cols, num_cols = clean_and_engineer_features(raw_df, use_accident_type=False)
    features = cat_cols + num_cols
    print(f"Valid records after datetime cleaning: {len(df):,}")
    print(f"Features used ({len(features)}): Categorical={len(cat_cols)}, Numeric={len(num_cols)}")

    # 3. Temporal Split
    train_df, test_df = temporal_split(df)
    print(f"Train split (2019-2021): {len(train_df):,} rows (Fatal rate: {train_df[TARGET].mean():.1%})")
    print(f"Test split  (2022):      {len(test_df):,} rows (Fatal rate: {test_df[TARGET].mean():.1%})")

    # 4. Group Rare Categories (< 100 in train)
    train_clean, test_clean, _ = group_rare_categories(train_df, test_df, cat_cols, min_count=100)

    # 5. Define Model Pipelines
    def prep_fn():
        return build_pipeline_preprocessor(cat_cols, num_cols)

    models = build_models(prep_fn, seed=args.seed)

    # 6. Validation Tuning & Test Evaluation
    print(f"\nTuning decision thresholds on 2021 validation set (Target Recall >= {args.target_recall:.2f})...")
    val_table, test_table, best_model, thresholds, fitted_models, test_probas = run_full_evaluation(
        train=train_clean,
        test=test_clean,
        features=features,
        target_col=TARGET,
        models_dict=models,
        target_recall=args.target_recall,
        seed=args.seed,
    )

    print("\n" + "=" * 50)
    print(" Validation Results (2021)")
    print("=" * 50)
    print(val_table[["pr_auc", "threshold", "recall", "precision", "als_dispatch_rate_pct"]].to_string())
    print(f"\n>> Selected Best Model (Highest PR-AUC): {best_model} (Threshold: {thresholds[best_model]:.3f})")

    print("\n" + "=" * 50)
    print(" Held-Out Test Evaluation (2022)")
    print("=" * 50)
    print(test_table[["recall", "precision", "accuracy", "als_dispatch_rate_pct", "tp", "fn", "fp", "pr_auc"]].to_string())

    # 7. Save Results
    eval_csv_path = os.path.join(args.output_dir, "model_evaluation_test2022.csv")
    test_table.reset_index().to_csv(eval_csv_path, index=False, encoding="utf-8")
    print(f"\n[Saved] Test evaluation table -> {eval_csv_path}")

    summary_data = {
        "total_records": len(df),
        "train_records_2019_2021": len(train_clean),
        "test_records_2022": len(test_clean),
        "overall_fatal_rate": float(df[TARGET].mean()),
        "train_fatal_rate": float(train_clean[TARGET].mean()),
        "test_fatal_rate": float(test_clean[TARGET].mean()),
        "best_model": best_model,
        "selected_threshold": float(thresholds[best_model]),
        "target_recall": float(args.target_recall),
        "test_metrics": test_table.reset_index().to_dict(orient="records"),
    }
    summary_json_path = os.path.join(args.output_dir, "summary_metrics.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Summary metrics JSON -> {summary_json_path}")

    # 8. Generate Visualizations
    bar_fig_path = os.path.join(args.fig_dir, "baseline_vs_models_recall.png")
    plot_baseline_vs_models(test_table, best_model, args.target_recall, bar_fig_path)
    print(f"[Saved] Model comparison chart -> {bar_fig_path}")

    tradeoff_fig_path = os.path.join(args.fig_dir, "tradeoff_als_vs_fatal_caught.png")
    plot_tradeoff_curve(
        y_test=test_clean[TARGET].values,
        best_proba=test_probas[best_model],
        chosen_stats=test_table.loc[best_model],
        rule_stats=test_table.loc["Baseline 2: Simple Rule (Risky Vehicle / Night)"],
        output_path=tradeoff_fig_path,
    )
    print(f"[Saved] Operational trade-off chart -> {tradeoff_fig_path}")

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
