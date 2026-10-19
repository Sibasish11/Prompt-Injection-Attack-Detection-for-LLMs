"""
Feature Engineering Module for Prompt Injection Detection.

This module combines two complementary feature representations:
1. N-Gram TF-IDF Vectorization:
   - Captures contextual lexical patterns and sub-phrases across unigrams, bigrams,
     and trigrams (e.g., 'ignore previous instructions', 'system prompt', 'developer mode').
2. Domain-Specific Handcrafted Heuristics:
   - Quantifies structural, syntactic, and adversarial signals that raw n-grams might miss,
     such as keyword frequencies, special character densities, delimiter frequency,
     base64 obfuscation patterns, and non-ASCII multilingual markers.

Design Rationale (Viva / Defense Notes):
----------------------------------------
- Why Hybrid Features? Pure TF-IDF struggles when attackers use obfuscation, novel phrasing,
  or non-standard syntactic delimiters (`---`, `###`, base64). Handcrafted statistical and
  rule-based features provide orthogonal inductive bias, allowing classical linear and tree-based
  classifiers to catch structural attacks even when unfamiliar n-grams appear.
- MaxAbsScaler for Sparse Stacking: Dense handcrafted features are scaled using `MaxAbsScaler`
  so their magnitudes align with unit-norm TF-IDF scores without densifying the sparse matrix
  (preserving O(N) memory efficiency).
"""

import os
import re
from typing import List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
import joblib


# Comprehensive taxonomy of prompt injection and jailbreak trigger phrases
INJECTION_KEYWORDS = [
    # Instruction Override & Negation
    "ignore", "disregard", "forget", "override", "bypass", "cancel", "stop",
    "ignore all", "ignore previous", "disregard previous", "forget instructions",
    "do not follow", "overwrite", "subvert",
    
    # System Prompt & Leakage Probes
    "system prompt", "system message", "initial prompt", "core instructions",
    "repeat the words above", "reveal", "print above", "output your prompt",
    "show instructions", "what are your instructions", "leak", "hidden prompt",
    "verbatim", "echo the prompt", "display prompt",
    
    # Persona & Role Hijacking (DAN, Developer Mode, Unrestricted)
    "pretend", "you are now", "act as", "roleplay as", "developer mode",
    "dan mode", "do anything now", "unrestricted", "unfiltered", "evil confidant",
    "jailbreak", "maximum", "hypothetical scenario", "jailbroken", "simulate",
    "assume the role", "no ethical guidelines", "without restrictions",
    
    # Structural & Authority Commands
    "administrative access", "root access", "sudo", "debug mode", "maintenance mode",
    "maintenance command", "api mode", "unlimited access", "bypass filter",
    "safety guidelines", "refusal policy", "safety policy", "disable safety",
    
    # Multilingual Injection Triggers (French, Spanish, German, Italian, etc.)
    "ignorez", "olvida", "ignora", "vergiss", "systeme", "instrucciones",
    "anweisungen", "directrices", "mode sans restriction"
]

# Delimiter markers commonly used in prompt framing and escape attacks
DELIMITER_PATTERNS = [
    r"---", r"===", r"###", r"```", r">>>", r"<<<",
    r"\[INST\]", r"\[/INST\]", r"<\|im_start\|>", r"<\|im_end\|>",
    r"<system>", r"</system>", r"\[SYSTEM\]"
]

def is_base64_payload(token: str) -> bool:
    """Detects realistic base64 encoded payload strings while avoiding false positives on long words."""
    if len(token) < 16:
        return False
    # Explicit base64 padding check
    if re.fullmatch(r"[A-Za-z0-9+/]{12,}={1,2}", token):
        return True
    # Contains base64 special chars (+ or /)
    if ("+" in token or "/" in token) and re.fullmatch(r"[A-Za-z0-9+/]+", token):
        return True
    # High entropy mixed case + digit block of length >= 24
    if len(token) >= 24 and re.fullmatch(r"[A-Za-z0-9+/]+", token):
        has_upper = any(c.isupper() for c in token)
        has_lower = any(c.islower() for c in token)
        has_digit = any(c.isdigit() for c in token)
        if has_upper and has_lower and has_digit:
            return True
    return False


def find_base64_candidates(text: str) -> List[str]:
    """Finds all potential base64 tokens in text."""
    tokens = re.findall(r"[A-Za-z0-9+/=]{16,}", text)
    return [t for t in tokens if is_base64_payload(t)]


class HandcraftedFeatureExtractor:
    """
    Extracts numerical and boolean domain-specific features from prompt text.
    """

    def __init__(self):
        self.feature_names = [
            "char_length",
            "word_count",
            "avg_word_length",
            "special_char_density",
            "uppercase_ratio",
            "punctuation_count",
            "imperative_keyword_count",
            "keyword_density",
            "delimiter_syntax_count",
            "non_ascii_ratio",
            "is_multilingual_flag",
            "base64_pattern_flag"
        ]

    def _extract_single(self, text: str) -> np.ndarray:
        """Extracts handcrafted feature vector for a single prompt string."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return np.zeros(len(self.feature_names), dtype=np.float32)

        raw_len = len(text)
        words = text.split()
        word_count = len(words)
        text_lower = text.lower()

        # 1. Length features
        char_len = float(raw_len)
        w_count = float(word_count)
        avg_word_len = float(raw_len / max(word_count, 1))

        # 2. Character composition features
        special_chars = len(re.findall(r"[^a-zA-Z0-9\s]", text))
        special_char_density = float(special_chars / max(raw_len, 1))
        
        uppercase_chars = sum(1 for c in text if c.isupper())
        uppercase_ratio = float(uppercase_chars / max(raw_len, 1))
        
        punctuation_count = float(len(re.findall(r"[!?,:;.\-\"\'\(\)\[\]\{\}]", text)))

        # 3. Imperative and injection trigger keyword frequencies
        keyword_hits = 0
        for kw in INJECTION_KEYWORDS:
            if " " in kw:
                keyword_hits += text_lower.count(kw)
            else:
                # Use word-boundary search for single words to avoid substring false positives
                keyword_hits += len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
        
        imperative_keyword_count = float(keyword_hits)
        keyword_density = float(keyword_hits / max(word_count, 1))

        # 4. Delimiter and structural escape syntax count
        delimiter_hits = 0
        for pat in DELIMITER_PATTERNS:
            delimiter_hits += len(re.findall(pat, text, re.IGNORECASE))
        delimiter_syntax_count = float(delimiter_hits)

        # 5. Multilingual and Unicode anomalies
        non_ascii_chars = sum(1 for c in text if ord(c) > 127)
        non_ascii_ratio = float(non_ascii_chars / max(raw_len, 1))
        is_multilingual_flag = 1.0 if non_ascii_chars > 3 else 0.0

        # 6. Base64 / Obfuscation pattern detection
        b64_matches = find_base64_candidates(text)
        has_b64 = 1.0 if len(b64_matches) > 0 else 0.0

        return np.array([
            char_len,
            w_count,
            avg_word_len,
            special_char_density,
            uppercase_ratio,
            punctuation_count,
            imperative_keyword_count,
            keyword_density,
            delimiter_syntax_count,
            non_ascii_ratio,
            is_multilingual_flag,
            has_b64
        ], dtype=np.float32)

    def transform(self, texts: Union[List[str], pd.Series, np.ndarray]) -> np.ndarray:
        """
        Transforms an iterable of raw text strings into a 2D numpy array of handcrafted features.
        """
        features_list = [self._extract_single(str(t)) for t in texts]
        return np.vstack(features_list)

    def get_feature_names(self) -> List[str]:
        return self.feature_names


class PromptInjectionFeaturePipeline:
    """
    Unified Feature Pipeline combining:
    1. Word-level N-gram TF-IDF (1-3 grams, max 5,000 features).
    2. Scaled Handcrafted Features via MaxAbsScaler.

    Provides end-to-end fit, transform, and serialization capabilities.
    """

    def __init__(self, max_tfidf_features: int = 5000):
        self.max_tfidf_features = max_tfidf_features
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=max_tfidf_features,
            sublinear_tf=True,
            min_df=2,
            strip_accents=None,  # Keep accents to preserve multilingual signals
            token_pattern=r"(?u)\b\w+\b|[#\-\=\`]{3,}"  # Capture words and structural delimiter tokens
        )
        self.handcrafted_extractor = HandcraftedFeatureExtractor()
        self.scaler = MaxAbsScaler()
        self.is_fitted = False

    def fit(self, texts: Union[List[str], pd.Series]) -> "PromptInjectionFeaturePipeline":
        """Fits the TF-IDF vectorizer and MaxAbsScaler on training texts."""
        print(f"[*] Fitting TF-IDF Vectorizer (max_features={self.max_tfidf_features}, ngrams=(1,3))...")
        self.tfidf.fit(texts)
        
        print("[*] Extracting and scaling handcrafted domain features...")
        handcrafted_matrix = self.handcrafted_extractor.transform(texts)
        self.scaler.fit(handcrafted_matrix)
        
        self.is_fitted = True
        print(f"[+] Feature pipeline successfully fitted with {len(self.get_feature_names()):,} total features.")
        return self

    def transform(self, texts: Union[List[str], pd.Series]) -> sp.csr_matrix:
        """
        Transforms input texts into a stacked sparse feature matrix (TF-IDF + Scaled Handcrafted).
        """
        if not self.is_fitted:
            raise ValueError("FeaturePipeline must be fitted before calling transform().")

        # 1. Transform TF-IDF
        tfidf_sparse = self.tfidf.transform(texts)

        # 2. Extract and scale handcrafted features
        handcrafted_raw = self.handcrafted_extractor.transform(texts)
        handcrafted_scaled = self.scaler.transform(handcrafted_raw)
        handcrafted_sparse = sp.csr_matrix(handcrafted_scaled)

        # 3. Concatenate horizontally as sparse matrix
        combined = sp.hstack([tfidf_sparse, handcrafted_sparse], format="csr")
        return combined

    def fit_transform(self, texts: Union[List[str], pd.Series]) -> sp.csr_matrix:
        """Fits and transforms input texts in one step."""
        return self.fit(texts).transform(texts)

    def get_feature_names(self) -> List[str]:
        """Returns ordered list of all feature names in the concatenated representation."""
        tfidf_names = list(self.tfidf.get_feature_names_out())
        handcrafted_names = [f"hc_{name}" for name in self.handcrafted_extractor.get_feature_names()]
        return tfidf_names + handcrafted_names

    def explain_prompt(self, text: str) -> dict:
        """
        Extracts human-interpretable feature triggers for explainability in CLI output.
        """
        text_lower = text.lower()
        matched_keywords = [kw for kw in INJECTION_KEYWORDS if (
            (" " in kw and kw in text_lower) or 
            (" " not in kw and re.search(r"\b" + re.escape(kw) + r"\b", text_lower))
        )]
        
        b64_matches = find_base64_candidates(text)
        delims = [pat for pat in DELIMITER_PATTERNS if re.search(pat, text, re.IGNORECASE)]
        non_ascii_count = sum(1 for c in text if ord(c) > 127)

        return {
            "matched_keywords": matched_keywords,
            "has_base64": len(b64_matches) > 0,
            "base64_snippets": b64_matches[:2],
            "matched_delimiters": delims,
            "non_ascii_count": non_ascii_count,
            "char_length": len(text),
            "word_count": len(text.split())
        }

    def save(self, file_path: str = "models/vectorizer.joblib") -> None:
        """Serializes the fitted pipeline to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)
        print(f"[+] Feature pipeline successfully saved -> '{file_path}'")

    @classmethod
    def load(cls, file_path: str = "models/vectorizer.joblib") -> "PromptInjectionFeaturePipeline":
        """Loads a pre-fitted feature pipeline from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fitted pipeline file not found at '{file_path}'. Run training first.")
        pipeline = joblib.load(file_path)
        return pipeline


def build_and_save_features(
    train_csv: str = "data/train.csv",
    test_csv: str = "data/test.csv",
    save_path: str = "models/vectorizer.joblib"
) -> Tuple[PromptInjectionFeaturePipeline, sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Utility function to build and fit the feature pipeline on training data and transform test data.
    """
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    pipeline = PromptInjectionFeaturePipeline(max_tfidf_features=5000)
    X_train = pipeline.fit_transform(train_df["cleaned_text"])
    X_test = pipeline.transform(test_df["cleaned_text"])
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    pipeline.save(save_path)
    return pipeline, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    build_and_save_features()
