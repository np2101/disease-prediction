"""
parkinsons_prediction.py — Parkinson's Disease Prediction
Dataset: UCI Parkinson's Dataset (voice measurements)
Target : 1 = Parkinson's, 0 = Healthy
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import compare_models, evaluate_model, save_model, plot_confusion_matrix, plot_feature_importance

MODEL_PATH   = "models/parkinsons_model.pkl"
RANDOM_STATE = 42

FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]


def load_data() -> tuple:
    """Load UCI Parkinson's dataset."""
    print("Loading UCI Parkinson's dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    try:
        df = pd.read_csv(url)
        X  = df[FEATURES].astype(float)
        y  = df["status"].astype(int)
    except Exception:
        print("  (Using synthetic demo data — check internet connection for real dataset)")
        np.random.seed(RANDOM_STATE)
        n  = 195
        X  = pd.DataFrame(
            np.random.randn(n, len(FEATURES)) * 50 + 100,
            columns=FEATURES
        )
        y  = pd.Series(np.random.randint(0, 2, n))

    print(f"  Samples: {len(X)} | Features: {X.shape[1]} | Positive rate: {y.mean():.2%}")
    return X, y


def get_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(probability=True, kernel="rbf", C=10, random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ]),
    }


def train() -> tuple:
    print("\n" + "=" * 50)
    print("   PARKINSON'S PREDICTION — TRAINING")
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
                          title="Parkinson's — Confusion Matrix",
                          save_path="models/parkinsons_confusion_matrix.png")

    save_model(best_model, MODEL_PATH)
    print(f"\n✓ Parkinson's model trained | Accuracy: {results['accuracy']:.4f}")
    return best_model, results


def predict_sample(model, sample: dict) -> dict:
    df    = pd.DataFrame([sample])[FEATURES]
    pred  = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    return {
        "prediction":  "PARKINSON'S DETECTED 🔴" if pred == 1 else "HEALTHY 🟢",
        "confidence":  round(float(max(proba)) * 100, 2),
        "probability": round(float(proba[1]) * 100, 2),
    }


if __name__ == "__main__":
    train()
