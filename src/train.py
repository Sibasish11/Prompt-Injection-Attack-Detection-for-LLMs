import os
import json
import time
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from src.features import PromptInjectionFeaturePipeline


def get_model_definitions(random_state: int = 42) -> Dict[str, Any]:
    """
    Initializes and returns dictionary of the 5 candidate ML classifiers and validates.
    """
    return {
        "logistic_regression": LogisticRegression(
            C=1.5,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state
        ),
        "linear_svm": CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                penalty="l2",
                loss="squared_hinge",
                dual=False,
                max_iter=2000,
                random_state=random_state
            ),
            cv=3
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=35,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1
        ),
        "multinomial_nb": MultinomialNB(
            alpha=0.1,
            fit_prior=True
        ),
        "xgboost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
    }


def train_and_cross_validate(
    train_csv: str = "data/train.csv",
    test_csv: str = "data/test.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Parameters:
        train_csv (str): Path to data/train.csv
        test_csv (str): Path to data/test.csv
        models_dir (str): Destination directory for trained artifacts.
        reports_dir (str): Destination directory for training summary tables.
        random_state (int): Random seed.

    Returns:
        Dict[str, Any]: Summary dictionary containing CV metrics and best model metadata.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 70)
    print("      STAGE: FEATURE EXTRACTION & PIPELINE FITTING")
    print("=" * 70)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # 1. Fit feature pipeline on training set
    feature_pipeline = PromptInjectionFeaturePipeline(max_tfidf_features=5000)
    X_train = feature_pipeline.fit_transform(train_df["cleaned_text"])
    y_train = train_df["label"].values

    # Save feature pipeline
    pipeline_path = os.path.join(models_dir, "vectorizer.joblib")
    feature_pipeline.save(pipeline_path)

    print("\n" + "=" * 70)
    print("      STAGE: 5-FOLD CROSS-VALIDATION & MODEL TRAINING")
    print("=" * 70)

    models = get_model_definitions(random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    cv_results_summary = []
    best_model_name = None
    best_cv_f1 = -1.0

    for name, model in models.items():
        print(f"\n[*] Evaluating '{name}' with 5-Fold Stratified Cross-Validation...")
        start_time = time.time()
        
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        elapsed_cv = time.time() - start_time

        mean_acc = np.mean(cv_scores["test_accuracy"])
        mean_prec = np.mean(cv_scores["test_precision"])
        mean_rec = np.mean(cv_scores["test_recall"])
        mean_f1 = np.mean(cv_scores["test_f1"])
        mean_auc = np.mean(cv_scores["test_roc_auc"])

        print(f"    -> Mean Accuracy:  {mean_acc * 100:.2f}%")
        print(f"    -> Mean Precision: {mean_prec * 100:.2f}%")
        print(f"    -> Mean Recall:    {mean_rec * 100:.2f}%")
        print(f"    -> Mean F1-Score:  {mean_f1 * 100:.2f}%")
        print(f"    -> Mean ROC-AUC:   {mean_auc:.4f}")
        print(f"    -> CV Completed in: {elapsed_cv:.2f}s")

        # Train model on full training partition
        print(f"[*] Fitting '{name}' on full training set ({X_train.shape[0]:,} samples)...")
        fit_start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - fit_start

        # Save individual model artifact
        model_save_path = os.path.join(models_dir, f"{name}.joblib")
        joblib.dump(model, model_save_path)
        print(f"[+] Saved '{name}' artifact -> '{model_save_path}' (fit time: {fit_time:.2f}s)")

        cv_results_summary.append({
            "model_name": name,
            "cv_accuracy": round(float(mean_acc), 4),
            "cv_precision": round(float(mean_prec), 4),
            "cv_recall": round(float(mean_rec), 4),
            "cv_f1": round(float(mean_f1), 4),
            "cv_roc_auc": round(float(mean_auc), 4),
            "train_fit_time_sec": round(float(fit_time), 2)
        })

        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_model_name = name

    # Convert CV results to DataFrame and save
    cv_summary_df = pd.DataFrame(cv_results_summary).sort_values(by="cv_f1", ascending=False)
    cv_csv_path = os.path.join(reports_dir, "cv_results.csv")
    cv_summary_df.to_csv(cv_csv_path, index=False)

    # Save best model copy as best_model.joblib for seamless CLI inference
    best_model_obj = models[best_model_name]
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model_obj, best_model_path)

    metadata = {
        "best_model_name": best_model_name,
        "best_cv_f1": round(float(best_cv_f1), 4),
        "models_evaluated": list(models.keys()),
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(models_dir, "best_model_meta.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 70)
    print("                5-FOLD CROSS-VALIDATION SUMMARY TABLE")
    print("=" * 70)
    print(cv_summary_df.to_string(index=False))
    print(f"\n[BEST MODEL] Best Selected Model by CV F1-Score: '{best_model_name}' (F1 = {best_cv_f1*100:.2f}%)")
    print(f"[+] Best model serialized to '{best_model_path}'")

    return {
        "cv_summary": cv_summary_df.to_dict(orient="records"),
        "best_model_name": best_model_name,
        "best_cv_f1": best_cv_f1
    }


if __name__ == "__main__":
    train_and_cross_validate()
