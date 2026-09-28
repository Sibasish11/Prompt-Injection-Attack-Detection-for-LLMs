import os
import re
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# Set global visualization aesthetics for high-quality reports
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
sns.set_palette(["#2b5c8f", "#d95f02", "#7570b3", "#e7298a"])
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
})


def clean_text(text: str) -> str:
    """
    Cleans raw prompt text while preserving structural syntax and punctuation.

    Parameters:
        text (str): Raw input prompt string.

    Returns:
        str: Normalized text string.

    Explanation:
        - We replace multiple spaces, tabs, and carriage returns with a single whitespace.
        - We strip leading and trailing whitespace.
        - We convert to lowercase to standardize vocabulary representation.
        - We explicitly DO NOT remove punctuation or special symbols because characters like
          `---`, `###`, `\n\n`, and backticks are critical structural breakout artifacts.
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize multiple whitespace characters (spaces, tabs, newlines) into clean single space
    cleaned = re.sub(r"\s+", " ", text).strip()
    # Lowercase normalization for consistent vocabulary mapping
    cleaned = cleaned.lower()
    return cleaned


def generate_eda_plots(df: pd.DataFrame, reports_dir: str = "reports") -> None:
    """
    Generates and saves comprehensive Exploratory Data Analysis (EDA) plots to reports/.

    Generates:
        1. eda_class_balance.png: Class balance (Benign vs. Injection).
        2. eda_length_distribution.png: Character and word length distributions by label.
        3. eda_category_distribution.png: Category breakdown across benign and injection classes.
        4. eda_top_ngrams.png: Comparative top unigrams and bigrams per class.

    Parameters:
        df (pd.DataFrame): The raw or preprocessed DataFrame.
        reports_dir (str): Directory where plots should be saved.
    """
    os.makedirs(reports_dir, exist_ok=True)
    print(f"[*] Generating EDA visualizations in '{reports_dir}/'...")

    # Compute auxiliary text metrics for EDA
    df_eda = df.copy()
    df_eda["char_length"] = df_eda["text"].fillna("").apply(len)
    df_eda["word_count"] = df_eda["text"].fillna("").apply(lambda s: len(s.split()))
    df_eda["label_name"] = df_eda["label"].map({0: "Benign (0)", 1: "Injection (1)"})

    # -------------------------------------------------------------
    # 1. Class Balance Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {"Benign (0)": "#2b5c8f", "Injection (1)": "#d95f02"}
    counts = df_eda["label_name"].value_counts()
    
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=palette, ax=ax, legend=False)
    ax.set_title("Dataset Class Balance (Mirror Prompt Injection)")
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Sample Count")
    
    # Add count annotations above bars
    for i, count in enumerate(counts.values):
        pct = (count / len(df_eda)) * 100
        ax.text(i, count + 80, f"{count:,} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")
    
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, "eda_class_balance.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 2. Prompt Length & Word Count Distributions
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Character length KDE / Histogram
    sns.histplot(
        data=df_eda,
        x="char_length",
        hue="label_name",
        palette=palette,
        kde=True,
        element="step",
        bins=40,
        ax=axes[0]
    )
    axes[0].set_title("Prompt Character Length Distribution")
    axes[0].set_xlabel("Character Length")
    axes[0].set_ylabel("Density / Count")
    
    # Word count KDE / Histogram
    sns.histplot(
        data=df_eda,
        x="word_count",
        hue="label_name",
        palette=palette,
        kde=True,
        element="step",
        bins=40,
        ax=axes[1]
    )
    axes[1].set_title("Prompt Word Count Distribution")
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Density / Count")
    
    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, "eda_length_distribution.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 3. Category Distribution by Label
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_counts = df_eda.groupby(["category", "label_name"]).size().reset_index(name="count")
    
    sns.barplot(
        data=cat_counts,
        x="category",
        y="count",
        hue="label_name",
        palette=palette,
        ax=ax
    )
    ax.set_title("Distribution of Attack Categories Across Labels")
    ax.set_xlabel("Attack Category")
    ax.set_ylabel("Number of Prompts")
    ax.legend(title="Label")
    
    # Annotate values on bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{int(height)}",
                        (p.get_x() + p.get_width() / 2., height + 30),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylim(0, cat_counts["count"].max() * 1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, "eda_category_distribution.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 4. Top Distinctive N-grams per Class
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for idx, (label_val, label_title) in enumerate([(0, "Benign (0)"), (1, "Injection (1)")]):
        subset = df_eda[df_eda["label"] == label_val]["text"]
        # Use CountVectorizer with bi/tri-grams
        vec = CountVectorizer(ngram_range=(1, 2), stop_words="english", max_features=15)
        vec_fit = vec.fit_transform(subset)
        sum_words = vec_fit.sum(axis=0)
        words_freq = [(word, sum_words[0, i]) for word, i in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:15]
        
        ngram_df = pd.DataFrame(words_freq, columns=["ngram", "frequency"])
        color = "#2b5c8f" if label_val == 0 else "#d95f02"
        
        sns.barplot(
            data=ngram_df,
            x="frequency",
            y="ngram",
            color=color,
            ax=axes[idx]
        )
        axes[idx].set_title(f"Top 15 N-grams: {label_title}")
        axes[idx].set_xlabel("Frequency")
        axes[idx].set_ylabel("N-gram")

    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, "eda_top_ngrams.png"), dpi=300)
    plt.close(fig)

    print(f"[+] EDA visualizations successfully exported to '{reports_dir}/'")


def preprocess_and_split(
    input_csv: str = "data/mirror-prompt-injection.csv",
    train_output_csv: str = "data/train.csv",
    test_output_csv: str = "data/test.csv",
    reports_dir: str = "reports",
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw data, performs cleaning, generates EDA, and performs joint stratification split.

    Parameters:
        input_csv (str): Path to mirror-prompt-injection.csv.
        train_output_csv (str): Destination for training split.
        test_output_csv (str): Destination for testing split.
        reports_dir (str): Destination for EDA charts.
        test_size (float): Fraction reserved for test set (default 0.20 = 80/20 split).
        random_state (int): Seed for reproducible data partitioning.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Dataset not found at '{input_csv}'. Ensure the mirror dataset is placed in 'data/'."
        )

    print(f"[*] Loading raw dataset from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    print(f"[+] Loaded {len(df):,} total samples with columns: {list(df.columns)}")

    # Ensure required columns exist
    required_cols = {"text", "label", "category"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}. Found: {df.columns}")

    # Handle any nulls if present
    initial_len = len(df)
    df = df.dropna(subset=["text", "label", "category"]).copy()
    if len(df) < initial_len:
        print(f"[!] Dropped {initial_len - len(df)} rows with missing values.")

    # Apply text cleaning (whitespace strip, lowercasing, punctuation retention)
    print("[*] Normalizing prompt text...")
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Generate EDA plots prior to splitting to capture global distribution
    generate_eda_plots(df, reports_dir=reports_dir)

    # Construct joint stratification key: e.g., "0_ignore", "1_extraction", "0_persona"
    df["strat_key"] = df["label"].astype(str) + "_" + df["category"].astype(str)

    print(f"[*] Performing Stratified Train/Test Split ({int((1-test_size)*100)}/{int(test_size*100)})...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["strat_key"],
        random_state=random_state
    )

    # Clean up auxiliary stratification column from export files
    train_df_out = train_df.drop(columns=["strat_key"])
    test_df_out = test_df.drop(columns=["strat_key"])

    os.makedirs(os.path.dirname(train_output_csv), exist_ok=True)
    train_df_out.to_csv(train_output_csv, index=False)
    test_df_out.to_csv(test_output_csv, index=False)

    print(f"[+] Successfully saved train set ({len(train_df):,} rows) -> '{train_output_csv}'")
    print(f"[+] Successfully saved test set ({len(test_df):,} rows) -> '{test_output_csv}'")

    # Print summary verification of stratification
    print("\n[+] Verification of Joint Stratification Distribution:")
    train_dist = train_df.groupby(["category", "label"]).size().unstack(fill_value=0)
    test_dist = test_df.groupby(["category", "label"]).size().unstack(fill_value=0)
    print("Train Split Distribution:\n", train_dist)
    print("Test Split Distribution:\n", test_dist)

    return train_df, test_df


if __name__ == "__main__":
    preprocess_and_split()
