import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('stroke_rf_model.pkl')
    scaler = joblib.load('stroke_scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

model, scaler, model_columns = load_artifacts()

# --- 2. APP HEADER ---
st.title("🧠 Stroke Risk Prediction App")
st.write("Enter patient details below to assess the estimated risk of a stroke.")

# --- 3. USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=0, max_value=120, value=50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension (High Blood Pressure)", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease History", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married?", ["No", "Yes"])

with col2:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=100.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# --- 4. PREPROCESS INPUTS ---
if st.button("Predict Stroke Risk", type="primary"):
    
    # Format raw inputs to match notebook mappings
    input_data = {
        'age': age,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'gender': 1 if gender == "Male" else 0,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'ever_married': 1 if ever_married == "Yes" else 0,
        'work_type': work_type,
        'Residence_type': residence_type,
        'smoking_status': smoking_status
    }
    
    # Create DataFrame for the single patient
    df_patient = pd.DataFrame([input_data])
    
    # One-Hot Encode categorical variables using the same logic as the notebook
    df_patient = pd.get_dummies(df_patient, columns=['work_type', 'Residence_type', 'smoking_status'])
    
    # Ensure all columns from training are present in exactly the same order
    for col in model_columns:
        if col not in df_patient.columns:
            df_patient[col] = 0
            
    df_patient = df_patient[model_columns] # Reorder columns
    
    # Scale the numerical columns
    numeric_cols = ['age', 'avg_glucose_level', 'bmi']
    df_patient[numeric_cols] = scaler.transform(df_patient[numeric_cols])
    
    # --- 5. MAKE PREDICTION ---
    # Get probability of stroke (Class 1)
    stroke_prob = model.predict_proba(df_patient)[0][1]
    
    # Apply our custom 0.30 threshold
    CUSTOM_THRESHOLD = 0.30
    is_at_risk = stroke_prob >= CUSTOM_THRESHOLD
    
    # --- 6. DISPLAY RESULTS ---
    st.markdown("---")
    
    # Define our new thresholds
    HIGH_RISK_THRESHOLD = 0.30
    MODERATE_RISK_THRESHOLD = 0.15
    
    if stroke_prob >= HIGH_RISK_THRESHOLD:
        st.error(f"### ⚠️ High Risk Detected")
        st.write(f"The model estimates a **{stroke_prob * 100:.1f}%** probability of a stroke.")
        st.info("Recommendation: Please consult a healthcare professional immediately for a formal medical assessment.")
        
    elif stroke_prob >= MODERATE_RISK_THRESHOLD:
        st.warning(f"### 🟡 Moderate Risk Detected")
        st.write(f"The model estimates a **{stroke_prob * 100:.1f}%** probability of a stroke.")
        st.write("*(This is significantly higher than the baseline average of ~5%.)*")
        st.info("Recommendation: Consider discussing lifestyle changes and monitoring with your doctor.")
        
    else:
        st.success(f"### ✅ Low Risk")
        st.write(f"The model estimates a **{stroke_prob * 100:.1f}%** probability of a stroke.")