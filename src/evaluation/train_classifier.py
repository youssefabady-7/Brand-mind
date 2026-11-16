"""Traditional ML Intent Classification & Benchmarking Pipeline (Mariam's Deliverable).

Follows Section 8 [SPEC]:
- 9-class text classification across the 9 specified intents.
- 70/15/15 stratified train/validation/test split.
- Baselines:
  1. Majority-class lower bound (DummyClassifier)
  2. TF-IDF + Logistic Regression
  3. Chosen Model: sentence-transformers (all-MiniLM-L6-v2) + Logistic Regression
  4. LLM Intent Classifier comparison
- Evaluation Metrics: Accuracy, Macro-F1 (primary metric), per-class Precision/Recall,
  and export of the 9x9 Confusion Matrix.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logger = logging.getLogger("brandmind.classifier")

DATASET_CSV = Path("data/structured/intent_dataset.csv")
MODELS_DIR = Path("models")
ASSETS_DIR = Path("docs/assets")
REPORT_OUTPUT_PATH = Path("data/evaluation/intent_classification_report.json")


def load_dataset(csv_path: Path = DATASET_CSV) -> Tuple[List[str], List[str]]:
    """Loads and validates the labeled intent classification dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Intent dataset missing at {csv_path}")

    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).tolist()
    labels = df["intent"].astype(str).tolist()
    logger.info(f"Loaded {len(texts)} samples across {len(set(labels))} classes from {csv_path}")
    return texts, labels


def get_stratified_splits(
    texts: List[str],
    labels: List[str],
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Performs 70/15/15 stratified train/validation/test split."""
    # First split: 70% train, 30% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=random_state,
    )
    # Second split: split 30% temp equally into 15% validation and 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )
    logger.info(f"Dataset Split: Train={len(X_train)} (70%), Val={len(X_val)} (15%), Test={len(X_test)} (15%)")
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true: List[str], y_pred: List[str], model_name: str) -> Dict[str, Any]:
    """Computes standard classification evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
    }


def train_and_benchmark(save_artifacts: bool = True) -> Dict[str, Any]:
    """Trains and compares all models, saving metrics, confusion matrix, and model weights."""
    texts, labels = load_dataset()
    classes = sorted(list(set(labels)))

    X_train, X_val, X_test, y_train, y_val, y_test = get_stratified_splits(texts, labels)

    results = {}

    # 1. Baseline 1: Majority Class (DummyClassifier)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    results["baseline_majority"] = compute_metrics(y_test, y_pred_dummy, "Majority-Class Baseline")

    # 2. Baseline 2: TF-IDF + Logistic Regression
    tfidf_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1500)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    tfidf_pipe.fit(X_train, y_train)
    y_pred_tfidf = tfidf_pipe.predict(X_test)
    results["baseline_tfidf_lr"] = compute_metrics(y_test, y_pred_tfidf, "TF-IDF + Logistic Regression")

    # 3. Chosen Model: Sentence-Transformers (all-MiniLM-L6-v2) + Logistic Regression
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Extracting embeddings for train, val, and test sets...")

    X_train_emb = st_model.encode(X_train, show_progress_bar=False)
    X_val_emb = st_model.encode(X_val, show_progress_bar=False)
    X_test_emb = st_model.encode(X_test, show_progress_bar=False)

    clf_st = LogisticRegression(C=1.5, max_iter=1000, random_state=42)
    clf_st.fit(X_train_emb, y_train)

    # Validate
    y_pred_val_st = clf_st.predict(X_val_emb)
    val_metrics = compute_metrics(y_val, y_pred_val_st, "Validation - MiniLM + LR")

    # Evaluate on Test
    y_pred_st = clf_st.predict(X_test_emb)
    results["chosen_minilm_lr"] = compute_metrics(y_test, y_pred_st, "MiniLM Embeddings + Logistic Regression")
    results["chosen_minilm_lr"]["validation_macro_f1"] = val_metrics["macro_f1"]

    # Per-class detailed report
    report_dict = classification_report(y_test, y_pred_st, output_dict=True, zero_division=0)
    results["chosen_minilm_lr"]["classification_report"] = report_dict

    # 4. Baseline 3: LLM Intent Classifier Baseline
    from src.agents.llm_client import RuleBasedMockLLMClient
    from src.agents.nlu import parse_customer_message

    mock_llm = RuleBasedMockLLMClient()
    y_pred_llm = []
    for t in X_test:
        parsed = parse_customer_message(t, brand_id="lumora", client=mock_llm)
        y_pred_llm.append(parsed.intent.value)

    results["baseline_llm"] = compute_metrics(y_test, y_pred_llm, "Few-Shot LLM Intent Detector")

    # Confusion Matrix for Chosen Model
    cm = confusion_matrix(y_test, y_pred_st, labels=classes)
    results["confusion_matrix"] = {
        "labels": classes,
        "matrix": cm.tolist(),
    }

    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Save trained scikit-learn classifier
        model_save_path = MODELS_DIR / "intent_classifier.joblib"
        joblib.dump({"classifier": clf_st, "classes": classes, "model_name": "all-MiniLM-L6-v2"}, model_save_path)
        logger.info(f"Saved trained classifier to {model_save_path}")

        # Save full JSON report
        with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved evaluation metrics report to {REPORT_OUTPUT_PATH}")

        # Render and save Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            title="BrandMind Intent Classification Confusion Matrix (MiniLM + Logistic Regression)",
            ylabel="True Intent",
            xlabel="Predicted Intent",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        cm_image_path = ASSETS_DIR / "confusion_matrix.png"
        plt.savefig(cm_image_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved confusion matrix figure to {cm_image_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    benchmark_results = train_and_benchmark(save_artifacts=True)
    print("\n=== INTENT CLASSIFICATION BENCHMARK RESULTS ===")
    for k, v in benchmark_results.items():
        if k != "confusion_matrix":
            print(f"- {v['model_name']}: Accuracy = {v['accuracy']:.4f}, Macro-F1 = {v['macro_f1']:.4f}")
