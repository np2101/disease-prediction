"""
diabetes_prediction.py — Diabetes Prediction using Pima Indians Dataset
Dataset: Built into sklearn / auto-downloaded from OpenML
Target : 1 = Diabetic, 0 = Not Diabetic
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import compare_models, evaluate_model, save_model, plot_confusion_matrix, plot_feature_importance

MODEL_PATH = "models/diabetes_model.pkl"
RANDOM_STATE = 42

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]


def load_data() -> tuple:
    """Load Pima Indians Diabetes dataset from OpenML."""
    print("Loading Pima Indians Diabetes dataset...")
    try:
        dataset = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
        X = dataset.data[FEATURES].astype(float)
        y = (dataset.target == "tested_positive").astype(int)
    except Exception:
        # Fallback: generate representative synthetic data for demo
        print("  (Using synthetic demo data — install internet for real dataset)")
        np.random.seed(RANDOM_STATE)
        n = 768
        X = pd.DataFrame({
            "Pregnancies":              np.random.randint(0, 17, n),
            "Glucose":                  np.random.randint(70, 200, n),
            "BloodPressure":            np.random.randint(40, 122, n),
            "SkinThickness":            np.random.randint(0, 99, n),
            "Insulin":                  np.random.randint(0, 846, n),
            "BMI":                      np.round(np.random.uniform(18, 67, n), 1),
            "DiabetesPedigreeFunction": np.round(np.random.uniform(0.08, 2.42, n), 3),
            "Age":                      np.random.randint(21, 81, n),
        })
        y = pd.Series((X["Glucose"] > 140).astype(int))

    print(f"  Samples: {len(X)} | Features: {X.shape[1]} | Positive rate: {y.mean():.2%}")
    return X, y


def get_models() -> dict:
    """Return candidate models for comparison."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }


def train() -> tuple:
    """Train, evaluate, and save the best diabetes model."""
    print("\n" + "=" * 50)
    print("   DIABETES PREDICTION — TRAINING")
    print("=" * 50)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models       = get_models()
    best_model, best_name = compare_models(models, X_train, y_train, X_test, y_test)

    results = evaluate_model(best_model, X_test, y_test, model_name=best_name)

    os.makedirs("models", exist_ok=True)
    plot_confusion_matrix(y_test, results["predictions"],
                          title="Diabetes — Confusion Matrix",
                          save_path="models/diabetes_confusion_matrix.png")

    # Feature importance (tree-based models only)
    raw = best_model.named_steps.get("clf", best_model) if hasattr(best_model, "named_steps") else best_model
    if hasattr(raw, "feature_importances_"):
        plot_feature_importance(raw.feature_importances_, FEATURES,
                                title="Diabetes — Feature Importance",
                                save_path="models/diabetes_feature_importance.png")

    save_model(best_model, MODEL_PATH)
    print(f"\n✓ Diabetes model trained | Accuracy: {results['accuracy']:.4f}")
    return best_model, results


def predict_sample(model, sample: dict) -> dict:
    """Predict diabetes for a single patient record."""
    df    = pd.DataFrame([sample])[FEATURES]
    pred  = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    return {
        "prediction":  "DIABETIC 🔴" if pred == 1 else "NOT DIABETIC 🟢",
        "confidence":  round(float(max(proba)) * 100, 2),
        "probability": round(float(proba[1]) * 100, 2),
    }


if __name__ == "__main__":
    train()
