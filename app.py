import streamlit as st
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ─── Page Config ───────────────────────────────────────
st.set_page_config(page_title="Disease Prediction System", page_icon="🏥", layout="centered")

st.title("🏥 Disease Prediction System")
st.markdown("Predict **Diabetes**, **Heart Disease**, or **Parkinson's** from medical data.")

# ─── Sidebar ───────────────────────────────────────────
disease = st.sidebar.selectbox(
    "Select Disease to Predict",
    ["Diabetes", "Heart Disease", "Parkinson's Disease"]
)

# ══════════════════════════════════════════════════════
# 1. DIABETES
# ══════════════════════════════════════════════════════
if disease == "Diabetes":
    st.header("🩸 Diabetes Prediction")
    st.markdown("Dataset: Pima Indians Diabetes (UCI)")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies   = st.number_input("Pregnancies",           0, 20,  1)
        glucose       = st.number_input("Glucose Level",         0, 300, 120)
        blood_pressure= st.number_input("Blood Pressure",        0, 150, 70)
        skin_thickness= st.number_input("Skin Thickness",        0, 100, 20)
    with col2:
        insulin       = st.number_input("Insulin Level",         0, 900, 80)
        bmi           = st.number_input("BMI",                   0.0, 70.0, 25.0)
        dpf           = st.number_input("Diabetes Pedigree",     0.0, 3.0,  0.5)
        age           = st.number_input("Age",                   1,  120,  30)

    if st.button("🔍 Predict Diabetes"):
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split

        # Load and train quickly
        data = fetch_openml(name='diabetes', version=1, as_frame=True)
        X, y = data.data, (data.target == 'tested_positive').astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        input_data = scaler.transform([[pregnancies, glucose, blood_pressure,
                                         skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction] * 100

        if prediction == 1:
            st.error(f"⚠️ Result: **DIABETIC** | Confidence: {confidence:.1f}%")
        else:
            st.success(f"✅ Result: **NOT DIABETIC** | Confidence: {confidence:.1f}%")

# ══════════════════════════════════════════════════════
# 2. HEART DISEASE
# ══════════════════════════════════════════════════════
elif disease == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")
    st.markdown("Dataset: Cleveland Heart Disease (UCI)")

    col1, col2 = st.columns(2)
    with col1:
        age      = st.number_input("Age",              1,   120, 50)
        sex      = st.selectbox("Sex",                 ["Male", "Female"])
        cp       = st.selectbox("Chest Pain Type",     [0, 1, 2, 3])
        trestbps = st.number_input("Resting BP",       80,  200, 120)
        chol     = st.number_input("Cholesterol",      100, 600, 200)
        fbs      = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg  = st.selectbox("Resting ECG",         [0, 1, 2])
    with col2:
        thalach  = st.number_input("Max Heart Rate",   60,  250, 150)
        exang    = st.selectbox("Exercise Angina",     [0, 1])
        oldpeak  = st.number_input("ST Depression",    0.0, 7.0, 1.0)
        slope    = st.selectbox("Slope",               [0, 1, 2])
        ca       = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
        thal     = st.selectbox("Thal",                [0, 1, 2, 3])

    if st.button("🔍 Predict Heart Disease"):
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split

        data = fetch_openml(name='heart-c', version=1, as_frame=True)
        X = data.data.apply(pd.to_numeric, errors='coerce').fillna(0)
        y = (data.target != 'negative').astype(int)

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        sex_val = 1 if sex == "Male" else 0
        input_data = scaler.transform([[age, sex_val, cp, trestbps, chol,
                                         fbs, restecg, thalach, exang,
                                         oldpeak, slope, ca, thal]])
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction] * 100

        if prediction == 1:
            st.error(f"⚠️ Result: **HEART DISEASE DETECTED** | Confidence: {confidence:.1f}%")
        else:
            st.success(f"✅ Result: **NO HEART DISEASE** | Confidence: {confidence:.1f}%")

# ══════════════════════════════════════════════════════
# 3. PARKINSON'S
# ══════════════════════════════════════════════════════
elif disease == "Parkinson's Disease":
    st.header("🧠 Parkinson's Disease Prediction")
    st.markdown("Dataset: UCI Parkinson's Dataset")

    col1, col2 = st.columns(2)
    with col1:
        fo    = st.number_input("MDVP:Fo (Hz)",     80.0,  300.0, 150.0)
        fhi   = st.number_input("MDVP:Fhi (Hz)",    100.0, 600.0, 200.0)
        flo   = st.number_input("MDVP:Flo (Hz)",    60.0,  250.0, 110.0)
        jitter= st.number_input("MDVP:Jitter(%)",   0.0,   1.0,   0.005)
        shimmer=st.number_input("MDVP:Shimmer",     0.0,   1.0,   0.03)
    with col2:
        nhr   = st.number_input("NHR",              0.0,   1.0,   0.02)
        hnr   = st.number_input("HNR",              5.0,   40.0,  22.0)
        rpde  = st.number_input("RPDE",             0.0,   1.0,   0.5)
        dfa   = st.number_input("DFA",              0.0,   1.0,   0.7)
        ppe   = st.number_input("PPE",              0.0,   1.0,   0.2)

    if st.button("🔍 Predict Parkinson's"):
        import urllib.request, io
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        
        with urllib.request.urlopen(url) as response:
            raw = response.read().decode()
        
        df = pd.read_csv(io.StringIO(raw))
        df = df.drop(columns=['name'])
        X = df.drop(columns=['status'])
        y = df['status']

        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = SVC(probability=True, kernel='rbf', random_state=42)
        model.fit(X_train_scaled, y_train)

        # Use only first 10 features to match simplified inputs
        input_array = np.array([[fo, fhi, flo, jitter, 0.0, 0.0, 0.0,
                                   shimmer, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   nhr, hnr, rpde, dfa, 0.0, ppe, 0.0, 0.0, 0.0]])
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled)[0][prediction] * 100

        if prediction == 1:
            st.error(f"⚠️ Result: **PARKINSON'S DETECTED** | Confidence: {confidence:.1f}%")
        else:
            st.success(f"✅ Result: **NO PARKINSON'S** | Confidence: {confidence:.1f}%")

# ─── Footer ────────────────────────────────────────────
st.markdown("---")
st.markdown("⚠️ *This tool is for educational purposes only. Always consult a doctor.*")
st.markdown("Built by **Nidhi Prajapati** | [GitHub](https://github.com/np2101)")