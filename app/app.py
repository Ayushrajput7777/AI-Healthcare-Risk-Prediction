import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

# Bilkul sahi rasta
BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "models" / "heart_model.pkl"
model = pickle.load(open(MODEL_PATH, 'rb'))

st.title("AI Healthcare Risk Prediction ❤️")
st.write("Heart disease ka risk check karo")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
thalch = st.number_input("Max Heart Rate", 70, 220, 150)
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
ca = st.number_input("CA", 0, 4, 0)
cp = st.selectbox("Chest Pain Type", ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
fbs = st.selectbox("Fasting Blood Sugar > 120", ["False", "True"])
restecg = st.selectbox("Rest ECG", ['normal', 'st-t abnormality', 'lv hypertrophy'])
exang = st.selectbox("Exercise Angina", ["False", "True"])
slope = st.selectbox("Slope", ['upsloping', 'flat', 'downsloping'])
thal = st.selectbox("Thal", ['normal', 'fixed defect', 'reversible defect'])
dataset = st.selectbox("Dataset", ['Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach'])

if st.button("Predict Risk"):
    raw = pd.DataFrame([{'id': 1, 'age': age, 'sex': sex, 'dataset': dataset, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalch': thalch, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}])
    raw = pd.get_dummies(raw)
    raw.columns = raw.columns.str.replace('reversible', 'reversable')
    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in raw.columns:
            raw[col] = 0
    raw = raw[expected_cols]
    pred = model.predict(raw)[0]
    if pred == 1:
        st.error("⚠️ High Risk - Doctor se consult karo")
    else:
        st.success("✅ Low Risk - Healthy!")