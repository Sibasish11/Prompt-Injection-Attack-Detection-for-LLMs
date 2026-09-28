import sys
import os
import argparse
import time

from src.preprocess import preprocess_and_split
from src.train import train_and_cross_validate
from src.evaluate import evaluate_all_models
from src.predict import run_predict_cli


def cmd_preprocess(args):
    """Executes the preprocessing and EDA stage."""
    print("\n" + "=" * 70)
    print(" >>> STEP 1: PREPROCESSING & EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    preprocess_and_split(
        input_csv=args.input_csv,
        train_output_csv=args.train_csv,
        test_output_csv=args.test_csv,
        reports_dir=args.reports_dir,
        test_size=args.test_size,
        random_state=args.random_state
    )


def cmd_train(args):
    """Executes the feature engineering, 5-fold CV, and model training stage."""
    print("\n" + "=" * 70)
    print(" >>> STEP 2: FEATURE ENGINEERING & MODEL TRAINING (5-FOLD CV)")
    print("=" * 70)
    train_and_cross_validate(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        random_state=args.random_state
    )


def cmd_evaluate(args):
    """Executes the test set evaluation and error analysis stage."""
    print("\n" + "=" * 70)
    print(" >>> STEP 3: UNSEEN HOLD-OUT TEST EVALUATION & ERROR ANALYSIS")
    print("=" * 70)
    evaluate_all_models(
        test_csv=args.test_csv,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir
    )


def cmd_predict(args):
    """Executes single prompt inference or launches interactive shell."""
    run_predict_cli(
        prompt=args.text,
        interactive=args.interactive,
        models_dir=args.models_dir
    )


def cmd_all(args):
    """Runs the full pipeline end-to-end in order."""
    start_total = time.time()
    print("\n" + "#" * 75)
    print("   STARTING FULL END-TO-END PROMPT INJECTION CLASSIFICATION PIPELINE")
    print("#" * 75)
    
    cmd_preprocess(args)
    cmd_train(args)
    cmd_evaluate(args)
    
    total_time = time.time() - start_total
    print("\n" + "#" * 75)
    print(f" [OK] FULL PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print(f"     - Trained Models Saved:   '{args.models_dir}/'")
    print(f"     - Evaluation & EDA Plots: '{args.reports_dir}/'")
    print(f"     - Benchmark CSVs:         '{args.reports_dir}/model_comparison.csv'")
    print("#" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prompt Injection Attack Detection for LLMs Using ML Techniques",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Common argument defaults
    parser.add_argument("--input-csv", default="data/mirror-prompt-injection.csv", help="Path to raw mirror dataset")
    parser.add_argument("--train-csv", default="data/train.csv", help="Path to processed train split")
    parser.add_argument("--test-csv", default="data/test.csv", help="Path to processed test split")
    parser.add_argument("--models-dir", default="models", help="Directory for saved model binaries")
    parser.add_argument("--reports-dir", default="reports", help="Directory for plots and reports")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test split proportion")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")

    subparsers = parser.add_subparsers(dest="command", help="Pipeline subcommands")

    # Subcommand: preprocess
    sub_preprocess = subparsers.add_parser("preprocess", help="Run text normalization, EDA, and joint stratification")
    sub_preprocess.set_defaults(func=cmd_preprocess)

    # Subcommand: train
    sub_train = subparsers.add_parser("train", help="Run 5-fold CV and train all 5 ML models")
    sub_train.set_defaults(func=cmd_train)

    # Subcommand: evaluate
    sub_evaluate = subparsers.add_parser("evaluate", help="Evaluate models on test set and generate reports")
    sub_evaluate.set_defaults(func=cmd_evaluate)

    # Subcommand: predict
    sub_predict = subparsers.add_parser("predict", help="Classify a prompt or start interactive session")
    sub_predict.add_argument("text", nargs="?", default=None, help="The prompt string to classify")
    sub_predict.add_argument("-i", "--interactive", action="store_true", help="Launch interactive terminal REPL")
    sub_predict.set_defaults(func=cmd_predict)

    # Subcommand: all
    sub_all = subparsers.add_parser("all", help="Execute complete pipeline (preprocess -> train -> evaluate)")
    sub_all.set_defaults(func=cmd_all)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Execute matched subcommand
    args.func(args)


if __name__ == "__main__":
    main()
