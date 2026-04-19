"""
heart_disease_prediction.py — Heart Disease Prediction using Cleveland Dataset
Dataset: UCI Heart Disease (auto-downloaded via sklearn)
Target : 1 = Heart Disease, 0 = No Heart Disease
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

MODEL_PATH   = "models/heart_disease_model.pkl"
RANDOM_STATE = 42

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

FEATURE_LABELS = [
    "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
    "Fasting BS", "Resting ECG", "Max Heart Rate", "Exercise Angina",
    "ST Depression", "ST Slope", "Major Vessels", "Thal"
]


def load_data() -> tuple:
    """Load Cleveland Heart Disease dataset."""
    print("Loading Cleveland Heart Disease dataset...")
    try:
        dataset = fetch_openml(name="heart-disease-cleveland", version=1,
                               as_frame=True, parser="auto")
        X = dataset.data.astype(float)
        X.columns = FEATURES
        y = (dataset.target.astype(float) > 0).astype(int)
        X = X.fillna(X.median())
    except Exception:
        print("  (Using synthetic demo data)")
        np.random.seed(RANDOM_STATE)
        n = 303
        X = pd.DataFrame({
            "age":      np.random.randint(29, 77, n),
            "sex":      np.random.randint(0, 2, n),
            "cp":       np.random.randint(0, 4, n),
            "trestbps": np.random.randint(94, 200, n),
            "chol":     np.random.randint(126, 564, n),
            "fbs":      np.random.randint(0, 2, n),
            "restecg":  np.random.randint(0, 3, n),
            "thalach":  np.random.randint(71, 202, n),
            "exang":    np.random.randint(0, 2, n),
            "oldpeak":  np.round(np.random.uniform(0, 6.2, n), 1),
            "slope":    np.random.randint(0, 3, n),
            "ca":       np.random.randint(0, 4, n),
            "thal":     np.random.randint(0, 4, n),
        })
        y = pd.Series((X["age"] > 55).astype(int))

    print(f"  Samples: {len(X)} | Features: {X.shape[1]} | Positive rate: {y.mean():.2%}")
    return X, y


def get_models() -> dict:
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
            ("clf",    SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }


def train() -> tuple:
    print("\n" + "=" * 50)
    print("   HEART DISEASE PREDICTION — TRAINING")
    print("=" * 50)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = get_models()
    best_model, best_name = compare_models(models, X_train, y_train, X_test, y_test)
    results = evaluate_model(best_model, X_test, y_test, model_name=best_name)

    os.makedirs("models", exist_ok=True)
    plot_confusion_matrix(y_test, results["predictions"],
                          title="Heart Disease — Confusion Matrix",
                          save_path="models/heart_confusion_matrix.png")

    raw = best_model.named_steps.get("clf", best_model) if hasattr(best_model, "named_steps") else best_model
    if hasattr(raw, "feature_importances_"):
        plot_feature_importance(raw.feature_importances_, FEATURE_LABELS,
                                title="Heart Disease — Feature Importance",
                                save_path="models/heart_feature_importance.png")

    save_model(best_model, MODEL_PATH)
    print(f"\n✓ Heart disease model trained | Accuracy: {results['accuracy']:.4f}")
    return best_model, results


def predict_sample(model, sample: dict) -> dict:
    df    = pd.DataFrame([sample])[FEATURES]
    pred  = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    return {
        "prediction":  "HEART DISEASE DETECTED 🔴" if pred == 1 else "NO HEART DISEASE 🟢",
        "confidence":  round(float(max(proba)) * 100, 2),
        "probability": round(float(proba[1]) * 100, 2),
    }


if __name__ == "__main__":
    train()
