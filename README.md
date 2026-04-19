# 🏥 Disease Prediction System

A production-ready Machine Learning system that predicts **Diabetes**, **Heart Disease**, and **Parkinson's Disease** from patient medical data.

Built for placement portfolios — clean modular Python, real datasets, CLI interface, model comparison.

---

## 📌 Why This Project Stands Out

- 3 real-world medical datasets (UCI / Kaggle)
- Compares 4 ML algorithms per disease and picks the best automatically
- Clean Python engineering — modular, typed, reusable
- CLI tool for live predictions
- Saves and reloads trained models with joblib

---

## 📁 Project Structure

```
disease-prediction/
├── data/                          # Datasets (auto-loaded via sklearn/UCI)
├── models/                        # Saved .pkl model files
├── diabetes_prediction.py         # Diabetes model module
├── heart_disease_prediction.py    # Heart disease model module
├── parkinsons_prediction.py       # Parkinson's model module
├── train_all.py                   # Train all 3 models at once
├── predict.py                     # CLI prediction tool
├── utils.py                       # Shared helpers (metrics, plots, preprocessing)
├── requirements.txt
└── README.md
```

---

## 🧠 ML Models Used

| Disease | Dataset | Best Model | Accuracy |
|---------|---------|------------|----------|
| Diabetes | Pima Indians (UCI) | Random Forest | ~78% |
| Heart Disease | Cleveland (UCI) | Gradient Boosting | ~85% |
| Parkinson's | UCI Parkinson's | SVM | ~87% |

---

## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/disease-prediction.git
cd disease-prediction
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## 🚀 Run

**Train all 3 models:**
```bash
python train_all.py
```

**Predict for a patient:**
```bash
python predict.py --disease diabetes
python predict.py --disease heart
python predict.py --disease parkinsons
```

---

## 📊 Sample Output

```
========================================
   DIABETES PREDICTION SYSTEM
========================================
Patient Data: Glucose=148, BMI=33.6, Age=50

Model Comparison:
  Logistic Regression  : 76.2%
  Random Forest        : 78.1%  ✓ Best
  SVM                  : 75.4%
  Gradient Boosting    : 77.8%

Prediction  : DIABETIC 🔴
Confidence  : 82.4%
========================================
```

---

## 🛠 Requirements

Python 3.9+, scikit-learn, pandas, numpy, matplotlib, seaborn, joblib

---

## 📄 License

MIT License
