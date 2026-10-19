"""
Inference and Prediction Engine for Prompt Injection Detection.

This module loads the trained feature pipeline and best classification model to perform:
1. Single prompt classification with confidence calibration.
2. Feature-level explainability (heuristic keyword hits, structural triggers, and TF-IDF terms).
3. Interactive REPL loop in the terminal for real-time security probing.
"""

import os
import json
import re
from typing import Dict, Any, Optional
import numpy as np
import joblib

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLOR_ENABLED = True
except ImportError:
    COLOR_ENABLED = False
    class Fore:
        RED = ""
        GREEN = ""
        YELLOW = ""
        CYAN = ""
        MAGENTA = ""
        WHITE = ""
        RESET = ""
    class Style:
        BRIGHT = ""
        RESET_ALL = ""

from src.features import PromptInjectionFeaturePipeline
from src.preprocess import clean_text


class PromptInjectionPredictor:
    """
    Production-grade predictor for prompt injection classification.
    """

    def __init__(
        self,
        models_dir: str = "models",
        model_name: Optional[str] = None
    ):
        self.models_dir = models_dir
        self.pipeline_path = os.path.join(models_dir, "vectorizer.joblib")
        
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(
                f"Feature pipeline not found at '{self.pipeline_path}'. Please run training first."
            )

        # Load feature pipeline
        self.pipeline = PromptInjectionFeaturePipeline.load(self.pipeline_path)

        # Determine model to load
        if model_name is None:
            best_model_path = os.path.join(models_dir, "best_model.joblib")
            if os.path.exists(best_model_path):
                self.model_path = best_model_path
                self.model_name = "best_model"
            else:
                self.model_path = os.path.join(models_dir, "linear_svm.joblib")
                self.model_name = "linear_svm"
        else:
            self.model_path = os.path.join(models_dir, f"{model_name}.joblib")
            self.model_name = model_name

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at '{self.model_path}'.")

        self.model = joblib.load(self.model_path)

        # Attempt to read best model metadata
        meta_path = os.path.join(models_dir, "best_model_meta.json")
        self.real_model_name = self.model_name
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.real_model_name = json.load(f).get("best_model_name", self.model_name)

    def predict(self, prompt: str) -> Dict[str, Any]:
        """
        Classifies a single prompt string.

        Parameters:
            prompt (str): Raw input prompt.

        Returns:
            Dict[str, Any]: Detailed prediction metadata including:
                - label: "INJECTION" or "BENIGN"
                - label_id: 1 or 0
                - confidence: float percentage (e.g. 98.6)
                - injection_probability: float [0.0, 1.0]
                - explanation: human-readable reason
                - triggers: dictionary of matched keywords/delimiters
        """
        if not prompt or len(prompt.strip()) == 0:
            return {
                "label": "BENIGN",
                "label_id": 0,
                "confidence": 100.0,
                "injection_probability": 0.0,
                "explanation": "Empty prompt provided (default safe).",
                "triggers": {}
            }

        cleaned = clean_text(prompt)
        X_vec = self.pipeline.transform([cleaned])

        # Predict probability
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_vec)[0]
            prob_inj = float(probs[1])
            prob_benign = float(probs[0])
        elif hasattr(self.model, "decision_function"):
            df = float(self.model.decision_function(X_vec)[0])
            prob_inj = 1.0 / (1.0 + np.exp(-df))
            prob_benign = 1.0 - prob_inj
        else:
            pred = int(self.model.predict(X_vec)[0])
            prob_inj = 1.0 if pred == 1 else 0.0
            prob_benign = 1.0 - prob_inj

        # Decision threshold at 0.50
        is_injection = prob_inj >= 0.50
        label_str = "INJECTION" if is_injection else "BENIGN"
        label_id = 1 if is_injection else 0
        confidence = (prob_inj if is_injection else prob_benign) * 100.0

        # Extract explainability signals
        signals = self.pipeline.explain_prompt(prompt)
        explanation_parts = []

        if signals["matched_keywords"]:
            top_kws = ", ".join([f"'{k}'" for k in signals["matched_keywords"][:4]])
            explanation_parts.append(f"matched injection trigger phrases ({top_kws})")
        
        if signals["has_base64"]:
            explanation_parts.append("detected potential base64-encoded obfuscation block")

        if signals["matched_delimiters"]:
            delims = ", ".join(signals["matched_delimiters"][:3])
            explanation_parts.append(f"detected prompt escape/framing delimiters ({delims})")

        if signals["non_ascii_count"] > 10:
            explanation_parts.append(f"contains significant non-ASCII / multilingual characters ({signals['non_ascii_count']} chars)")

        if not explanation_parts:
            if is_injection:
                explanation_parts.append("statistical n-gram and vocabulary distribution aligns with known injection attacks")
            else:
                explanation_parts.append("natural linguistic syntax with no adversarial override patterns detected")

        explanation_str = "; ".join(explanation_parts)

        return {
            "prompt": prompt,
            "label": label_str,
            "label_id": label_id,
            "confidence": round(confidence, 2),
            "injection_probability": round(prob_inj, 4),
            "explanation": explanation_str,
            "triggers": signals,
            "model_used": self.real_model_name
        }

    def format_cli_output(self, result: Dict[str, Any]) -> str:
        """
        Formats the prediction result into a formatted terminal card.
        """
        is_inj = result["label"] == "INJECTION"
        badge_color = Fore.RED if is_inj else Fore.GREEN
        badge_text = f"[{result['label']}]"
        
        lines = [
            f"\n{Style.BRIGHT}{'=' * 65}",
            f" PROMPT INJECTION CLASSIFICATION RESULT",
            f"{'=' * 65}{Style.RESET_ALL}",
            f" Prediction:    {badge_color}{Style.BRIGHT}{badge_text}{Style.RESET_ALL}",
            f" Confidence:    {Style.BRIGHT}{result['confidence']:.2f}%{Style.RESET_ALL} (P(Injection) = {result['injection_probability']:.4f})",
            f" Active Model:  {result.get('model_used', 'N/A').replace('_', ' ').title()}",
            f" Explanation:   {Fore.CYAN}{result['explanation']}{Style.RESET_ALL}",
            f"{'=' * 65}"
        ]
        return "\n".join(lines)


def run_predict_cli(
    prompt: Optional[str] = None,
    interactive: bool = False,
    models_dir: str = "models"
) -> None:
    """
    CLI runner for prompt prediction and interactive shell.
    """
    predictor = PromptInjectionPredictor(models_dir=models_dir)

    if interactive:
        print(f"\n{Style.BRIGHT}{Fore.CYAN}{'=' * 70}")
        print("  PROMPT INJECTION DETECTOR - INTERACTIVE SECURITY TESTING REPL")
        print(f"  Model: {predictor.real_model_name.replace('_', ' ').title()} | Classical ML Inference (<2ms latency)")
        print(f"  Type your prompt to inspect. Type 'exit' or 'quit' to terminate.")
        print(f"{'=' * 70}{Style.RESET_ALL}\n")

        while True:
            try:
                user_input = input(f"{Style.BRIGHT}Prompt >> {Style.RESET_ALL}").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "q"):
                    print(f"\n{Fore.YELLOW}[*] Exiting interactive inspection session. Stay safe!{Style.RESET_ALL}")
                    break

                res = predictor.predict(user_input)
                print(predictor.format_cli_output(res))
                print()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{Fore.YELLOW}[*] Session aborted by user.{Style.RESET_ALL}")
                break
    else:
        if prompt is None or len(prompt.strip()) == 0:
            print("[!] Error: No prompt text provided. Usage: python main.py predict \"<prompt text>\"")
            return
        
        res = predictor.predict(prompt)
        print(predictor.format_cli_output(res))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--interactive", "-i"):
            run_predict_cli(interactive=True)
        else:
            run_predict_cli(prompt=" ".join(sys.argv[1:]))
    else:
        run_predict_cli(interactive=True)
