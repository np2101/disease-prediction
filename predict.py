"""
predict.py — CLI tool to predict disease for a sample patient.
Usage:
    python predict.py --disease diabetes
    python predict.py --disease heart
    python predict.py --disease parkinsons
"""

import argparse
from utils import load_model


# ── Sample patients (realistic test cases) ────────────────────────────────────

SAMPLES = {
    "diabetes": {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    },
    "heart": {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    },
    "parkinsons": {
        "MDVP:Fo(Hz)": 119.992, "MDVP:Fhi(Hz)": 157.302,
        "MDVP:Flo(Hz)": 74.997, "MDVP:Jitter(%)": 0.00784,
        "MDVP:Jitter(Abs)": 0.00007, "MDVP:RAP": 0.00370,
        "MDVP:PPQ": 0.00554, "Jitter:DDP": 0.01109,
        "MDVP:Shimmer": 0.04374, "MDVP:Shimmer(dB)": 0.42600,
        "Shimmer:APQ3": 0.02182, "Shimmer:APQ5": 0.03130,
        "MDVP:APQ": 0.02971, "Shimmer:DDA": 0.06545,
        "NHR": 0.02211, "HNR": 21.033, "RPDE": 0.414783,
        "DFA": 0.815285, "spread1": -4.813031, "spread2": 0.266482,
        "D2": 2.301442, "PPE": 0.284654,
    },
}

MODEL_PATHS = {
    "diabetes":   "models/diabetes_model.pkl",
    "heart":      "models/heart_disease_model.pkl",
    "parkinsons": "models/parkinsons_model.pkl",
}

TITLES = {
    "diabetes":   "DIABETES PREDICTION",
    "heart":      "HEART DISEASE PREDICTION",
    "parkinsons": "PARKINSON'S PREDICTION",
}


def run_prediction(disease: str) -> None:
    """Load model, run prediction, and print formatted results."""
    if disease not in MODEL_PATHS:
        print(f"Unknown disease '{disease}'. Choose: diabetes, heart, parkinsons")
        return

    print("\n" + "=" * 50)
    print(f"   {TITLES[disease]}")
    print("=" * 50)

    model  = load_model(MODEL_PATHS[disease])
    sample = SAMPLES[disease]

    print("\nPatient Data:")
    for key, val in sample.items():
        print(f"  {key:<35} {val}")

    # Import the right predictor
    if disease == "diabetes":
        from diabetes_prediction import predict_sample
    elif disease == "heart":
        from heart_disease_prediction import predict_sample
    else:
        from parkinsons_prediction import predict_sample

    result = predict_sample(model, sample)

    print("\n" + "-" * 50)
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Confidence  : {result['confidence']}%")
    print(f"  Probability : {result['probability']}% chance of disease")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Disease Prediction System — predict from sample patient data."
    )
    parser.add_argument(
        "--disease",
        type=str,
        required=True,
        choices=["diabetes", "heart", "parkinsons"],
        help="Which disease to predict: diabetes | heart | parkinsons",
    )
    args = parser.parse_args()
    run_prediction(args.disease)


if __name__ == "__main__":
    main()
