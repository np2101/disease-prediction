"""
utils.py — Shared helper functions for all disease prediction modules.
Covers: preprocessing, model evaluation, plotting, and saving/loading models.
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_score


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(model: Any, path: str) -> None:
    """Save a trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved → {path}")


def load_model(path: str) -> Any:
    """Load a saved model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at '{path}'. Run train_all.py first.")
    return joblib.load(path)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Return accuracy, AUC, and print a classification report."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"\n── {model_name} ──────────────────────────")
    print(f"  Accuracy : {acc:.4f}")
    if auc:
        print(f"  AUC-ROC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    return {"accuracy": acc, "auc": auc, "predictions": y_pred}


def compare_models(models: dict, X_train, y_train, X_test, y_test) -> tuple[Any, str]:
    """
    Train and compare multiple models.
    Returns (best_model, best_model_name).
    """
    print("\n── Model Comparison (5-fold CV Accuracy) ──────────────")
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        results[name] = cv_scores.mean()
        marker = ""
        print(f"  {name:<25} {cv_scores.mean():.4f} ± {cv_scores.std():.4f} {marker}")

    best_name  = max(results, key=results.get)
    best_model = models[best_name]
    print(f"\n  ✓ Best model: {best_name} ({results[best_name]:.4f})")

    best_model.fit(X_train, y_train)
    return best_model, best_name


# ── Preprocessing ─────────────────────────────────────────────────────────────

def scale_features(X_train, X_test):
    """Standard-scale features. Returns (X_train_scaled, X_test_scaled, scaler)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, title: str, save_path: str) -> None:
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_feature_importance(importances: np.ndarray, feature_names: list,
                            title: str, save_path: str) -> None:
    """Bar chart of feature importances."""
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances[indices], color="steelblue")
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")
