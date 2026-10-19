"""
Model Evaluation and Error Analysis Module.

This module evaluates all trained models on the unseen hold-out test set (20% partition),
producing comprehensive security metrics, confusion matrices, ROC/PR curves, category-level
recall breakdowns, and granular false positive / false negative error analysis logs.

Key Security Evaluation Concepts (Viva Defense):
------------------------------------------------
1. Injection Class Recall (True Positive Rate):
   - In AI security guardrails, Recall is paramount: a False Negative (FN) means an attacker's
     malicious payload bypassed the filter to reach the LLM, potentially causing data leaks,
     instruction overrides, or jailbreaks.
2. Injection Class Precision (Alert Fidelity):
   - High Precision prevents user friction and false alarms on legitimate prompts that happen
     to discuss AI ethics, roleplay, or coding instructions.
3. Category-Specific Breakdown:
   - Evaluates whether the classifier performs equally well across diverse attack vectors
     (`ignore`, `extraction`, `persona`, `multilingual`). Multilingual and extraction attacks
     often exhibit unique syntactic distributions that standard keyword filters miss.
"""

import os
import json
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)

from src.features import PromptInjectionFeaturePipeline


# Configure plotting styles
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
})


def load_model_and_predict(
    model_path: str,
    X_test,
    model_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads model and produces discrete predictions and positive class probabilities.
    """
    model = joblib.load(model_path)
    preds = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # Map decision function to pseudo-probability via logistic sigmoid
        df = model.decision_function(X_test)
        probs = 1.0 / (1.0 + np.exp(-df))
    else:
        probs = preds.astype(float)

    return preds, probs


def plot_all_confusion_matrices(
    cm_dict: Dict[str, np.ndarray],
    reports_dir: str = "reports"
) -> None:
    """
    Plots and exports a cohesive multi-panel figure displaying confusion matrices for all 5 models.
    """
    model_names = list(cm_dict.keys())
    n_models = len(model_names)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for idx, name in enumerate(model_names):
        ax = axes_flat[idx]
        cm = cm_dict[name]
        
        # Calculate percentages
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annot_matrix = np.array([
            [f"{cm[0,0]:,}\n({cm_pct[0,0]:.1f}%)", f"{cm[0,1]:,}\n({cm_pct[0,1]:.1f}%)"],
            [f"{cm[1,0]:,}\n({cm_pct[1,0]:.1f}%)", f"{cm[1,1]:,}\n({cm_pct[1,1]:.1f}%)"]
        ])

        sns.heatmap(
            cm,
            annot=annot_matrix,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred Benign (0)", "Pred Injection (1)"],
            yticklabels=["True Benign (0)", "True Injection (1)"],
            annot_kws={"fontsize": 11, "fontweight": "bold"}
        )
        ax.set_title(f"Confusion Matrix: {name.replace('_', ' ').title()}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    # Hide extra unused subplot if odd number
    for j in range(n_models, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, "confusion_matrices.png"), dpi=300)
    plt.close(fig)
    print(f"[+] Multi-model confusion matrices saved -> '{os.path.join(reports_dir, 'confusion_matrices.png')}'")


def plot_roc_and_pr_curves(
    curves_data: Dict[str, Dict[str, np.ndarray]],
    reports_dir: str = "reports"
) -> None:
    """
    Plots Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. ROC Curves
    for name, data in curves_data.items():
        fpr, tpr, roc_auc = data["fpr"], data["tpr"], data["roc_auc"]
        axes[0].plot(fpr, tpr, lw=2, label=f"{name.replace('_', ' ').title()} (AUC = {roc_auc:.4f})")
    
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (0.50)")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall / Sensitivity)")
    axes[0].set_title("ROC Curves Comparison")
    axes[0].legend(loc="lower right")

    # 2. Precision-Recall Curves
    for name, data in curves_data.items():
        precision, recall, f1 = data["precision_curve"], data["recall_curve"], data["f1_score"]
        axes[1].plot(recall, precision, lw=2, label=f"{name.replace('_', ' ').title()} (F1 = {f1:.4f})")
    
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (True Positive Rate)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].set_title("Precision-Recall Curves Comparison")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, "roc_pr_curves.png"), dpi=300)
    plt.close(fig)
    print(f"[+] ROC and Precision-Recall curves saved -> '{os.path.join(reports_dir, 'roc_pr_curves.png')}'")


def evaluate_all_models(
    test_csv: str = "data/test.csv",
    models_dir: str = "models",
    reports_dir: str = "reports"
) -> pd.DataFrame:
    """
    Evaluates all saved models against the test set, performs error analysis and category breakdown.

    Parameters:
        test_csv (str): Path to data/test.csv
        models_dir (str): Path to directory holding trained models.
        reports_dir (str): Path to reports output directory.

    Returns:
        pd.DataFrame: Comparison table ranking all evaluated models.
    """
    os.makedirs(reports_dir, exist_ok=True)
    
    print("=" * 70)
    print("         STAGE: UNSEEN HOLD-OUT TEST SET EVALUATION")
    print("=" * 70)

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test split not found at '{test_csv}'. Run preprocessing first.")

    test_df = pd.read_csv(test_csv)
    pipeline_path = os.path.join(models_dir, "vectorizer.joblib")
    feature_pipeline = PromptInjectionFeaturePipeline.load(pipeline_path)

    X_test = feature_pipeline.transform(test_df["cleaned_text"])
    y_test = test_df["label"].values

    # Read best model metadata if exists
    meta_path = os.path.join(models_dir, "best_model_meta.json")
    best_model_name = "linear_svm"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            best_model_name = json.load(f).get("best_model_name", "linear_svm")

    model_files = [
        ("logistic_regression", "logistic_regression.joblib"),
        ("linear_svm", "linear_svm.joblib"),
        ("random_forest", "random_forest.joblib"),
        ("multinomial_nb", "multinomial_nb.joblib"),
        ("xgboost", "xgboost.joblib")
    ]

    metrics_list = []
    cm_dict = {}
    curves_dict = {}
    all_predictions = {}
    all_probabilities = {}

    for model_key, filename in model_files:
        model_path = os.path.join(models_dir, filename)
        if not os.path.exists(model_path):
            print(f"[!] Warning: Model file '{filename}' not found. Skipping.")
            continue

        preds, probs = load_model_and_predict(model_path, X_test, model_key)
        all_predictions[model_key] = preds
        all_probabilities[model_key] = probs

        # Compute Core Metrics
        acc = accuracy_score(y_test, preds)
        prec_inj = precision_score(y_test, preds, pos_label=1, zero_division=0)
        rec_inj = recall_score(y_test, preds, pos_label=1, zero_division=0)
        f1_inj = f1_score(y_test, preds, pos_label=1, zero_division=0)
        f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
        roc_auc = roc_auc_score(y_test, probs)

        cm = confusion_matrix(y_test, preds)
        cm_dict[model_key] = cm

        # Calculate ROC and PR curves data
        fpr, tpr, _ = roc_curve(y_test, probs)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs)
        curves_dict[model_key] = {
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "precision_curve": prec_curve,
            "recall_curve": rec_curve,
            "f1_score": f1_inj
        }

        metrics_list.append({
            "Model": model_key.replace("_", " ").title(),
            "Accuracy": round(acc * 100, 2),
            "Precision (Inj)": round(prec_inj * 100, 2),
            "Recall (Inj)": round(rec_inj * 100, 2),
            "F1-Score (Inj)": round(f1_inj * 100, 2),
            "Macro F1": round(f1_macro * 100, 2),
            "ROC-AUC": round(roc_auc, 4),
            "False Positives": int(cm[0, 1]),
            "False Negatives": int(cm[1, 0])
        })

    # Generate Comparison DataFrame
    comparison_df = pd.DataFrame(metrics_list).sort_values(by="F1-Score (Inj)", ascending=False)
    comparison_csv_path = os.path.join(reports_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)

    print("\n" + "=" * 85)
    print("                    MODEL BENCHMARK COMPARISON TABLE (TEST SET)")
    print("=" * 85)
    print(comparison_df.to_string(index=False))
    print(f"\n[+] Saved comparison metrics -> '{comparison_csv_path}'")

    # Plot Visualizations
    plot_all_confusion_matrices(cm_dict, reports_dir=reports_dir)
    plot_roc_and_pr_curves(curves_dict, reports_dir=reports_dir)

    # -------------------------------------------------------------
    # Category Breakdown for Best Model and All Models
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"      CATEGORY-WISE BREAKDOWN (Best Model: '{best_model_name}')")
    print("=" * 70)

    best_preds = all_predictions[best_model_name]
    best_probs = all_probabilities[best_model_name]

    test_df_eval = test_df.copy()
    test_df_eval["predicted_label"] = best_preds
    test_df_eval["injection_prob"] = best_probs

    category_records = []
    categories = sorted(test_df_eval["category"].unique())

    for cat in categories:
        cat_subset = test_df_eval[test_df_eval["category"] == cat]
        y_true_cat = cat_subset["label"].values
        y_pred_cat = cat_subset["predicted_label"].values

        total_cat = len(cat_subset)
        inj_count = int(np.sum(y_true_cat == 1))
        ben_count = int(np.sum(y_true_cat == 0))

        cat_acc = accuracy_score(y_true_cat, y_pred_cat)
        cat_rec = recall_score(y_true_cat, y_pred_cat, pos_label=1, zero_division=0)
        cat_prec = precision_score(y_true_cat, y_pred_cat, pos_label=1, zero_division=0)
        cat_f1 = f1_score(y_true_cat, y_pred_cat, pos_label=1, zero_division=0)

        category_records.append({
            "Category": cat,
            "Total Samples": total_cat,
            "Injections": inj_count,
            "Benign": ben_count,
            "Accuracy": round(cat_acc * 100, 2),
            "Precision (Inj)": round(cat_prec * 100, 2),
            "Recall (Inj)": round(cat_rec * 100, 2),
            "F1-Score (Inj)": round(cat_f1 * 100, 2)
        })

    category_df = pd.DataFrame(category_records).sort_values(by="Recall (Inj)", ascending=True)
    category_csv_path = os.path.join(reports_dir, "category_breakdown.csv")
    category_df.to_csv(category_csv_path, index=False)
    print(category_df.to_string(index=False))
    print(f"\n[+] Category breakdown saved -> '{category_csv_path}'")

    # -------------------------------------------------------------
    # Granular Error Analysis: False Positives & False Negatives
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("                     GRANULAR ERROR ANALYSIS")
    print("=" * 70)

    fp_mask = (test_df_eval["label"] == 0) & (test_df_eval["predicted_label"] == 1)
    fn_mask = (test_df_eval["label"] == 1) & (test_df_eval["predicted_label"] == 0)

    fp_df = test_df_eval[fp_mask].copy()
    fp_df["error_type"] = "False Positive (Benign marked as Injection)"

    fn_df = test_df_eval[fn_mask].copy()
    fn_df["error_type"] = "False Negative (Injection missed by model)"

    errors_df = pd.concat([fp_df, fn_df], ignore_index=True)
    errors_csv_path = os.path.join(reports_dir, "errors.csv")
    
    # Save selected informative columns
    export_cols = ["error_type", "category", "label", "predicted_label", "injection_prob", "text"]
    errors_df[export_cols].to_csv(errors_csv_path, index=False)

    print(f"[!] Total Prediction Errors on Test Set ({len(test_df_eval):,} samples): {len(errors_df):,}")
    print(f"    -> False Positives (Type I Error): {len(fp_df):,}")
    print(f"    -> False Negatives (Type II Error - Security Breaches): {len(fn_df):,}")
    print("\nErrors by Attack Category:")
    error_by_cat = errors_df.groupby(["category", "error_type"]).size().unstack(fill_value=0)
    print(error_by_cat)
    print(f"\n[+] Detailed error log dumped -> '{errors_csv_path}'")

    return comparison_df


if __name__ == "__main__":
    evaluate_all_models()
