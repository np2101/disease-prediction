"""
train_all.py — Train all 3 disease prediction models in one run.
Run: python train_all.py
"""

import time
from diabetes_prediction    import train as train_diabetes
from heart_disease_prediction import train as train_heart
from parkinsons_prediction  import train as train_parkinsons


def main():
    print("\n" + "=" * 50)
    print("   DISEASE PREDICTION SYSTEM — TRAINING ALL")
    print("=" * 50)

    start = time.time()
    results = {}

    # ── Train each model ──────────────────────────────────────────────────────
    _, r1 = train_diabetes()
    results["Diabetes"]      = r1["accuracy"]

    _, r2 = train_heart()
    results["Heart Disease"] = r2["accuracy"]

    _, r3 = train_parkinsons()
    results["Parkinson's"]   = r3["accuracy"]

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "=" * 50)
    print("   TRAINING COMPLETE — SUMMARY")
    print("=" * 50)
    for disease, acc in results.items():
        print(f"  {disease:<20} Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"\n  Total time: {elapsed:.1f}s")
    print("  Models saved to: models/")
    print("=" * 50)
    print("\nRun predictions with:")
    print("  python predict.py --disease diabetes")
    print("  python predict.py --disease heart")
    print("  python predict.py --disease parkinsons")


if __name__ == "__main__":
    main()
