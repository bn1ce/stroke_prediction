import streamlit as st
import pandas as pd
import joblib
import os
from tensorflow import keras

# --- 1. PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title="Stroke Risk Estimator", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* General spacing and background */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Header typography */
    .proto-label { font-size: 0.8rem; color: #8898aa; font-weight: bold; letter-spacing: 1px; margin-bottom: -10px; }
    .main-title { font-family: 'Georgia', serif; font-size: 3rem; color: #0f4c81; font-weight: bold; margin-bottom: 0px; padding-bottom: 0px; }
    .live-italic { color: #2ea673; font-style: italic; }
    
    /* Metrics row styling */
    div[data-testid="metric-container"] { background-color: white; border: 1px solid #e6e9ef; padding: 15px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- 2. HEADER & MODEL SELECTION ---
col_head_left, col_head_right = st.columns([6, 4])

with col_head_left:
    st.markdown('<div class="proto-label">PROTOTYPE — ACADEMIC USE ONLY</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Stroke risk, <span class="live-italic">estimated live</span></div>', unsafe_allow_html=True)

with col_head_right:
    st.write("") # Spacing
    model_choice = st.selectbox(
        "Select Model Architecture:", 
        ["Random Forest", "Support Vector Machine (SVM)", "Artificial Neural Network (ANN)"]
    )
    st.caption("Trained on: **healthcare-dataset-stroke-data.csv**<br>Class balancing: **SMOTE (train set only)**", unsafe_allow_html=True)

st.markdown("---")

# --- 3. LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts(model_name):
    try:
        if model_name == "Random Forest":
            model = joblib.load('stroke_rf_model.pkl')
            model_columns = joblib.load('model_columns.pkl') # RF's columns
            scaler = joblib.load('stroke_scaler.pkl')        # RF's scaler
            model_type = "sklearn"
            
        elif model_name == "Support Vector Machine (SVM)":
            model = joblib.load('stroke_svm_model.pkl')
            model_columns = joblib.load('svm_columns.pkl')   # SVM's columns
            scaler = None                                    # SVM pipeline handles its own scaling
            model_type = "sklearn"
            
        elif model_name == "Artificial Neural Network (ANN)":
            model = keras.models.load_model('stroke_ann_smote.keras')
            model_columns = joblib.load('ann_columns.pkl') # Placeholder: update if ANN differs
            scaler = joblib.load('ann_scaler.pkl')        # Placeholder
            model_type = "keras"

        return model, scaler, model_columns, model_type, "Success"
    except Exception as e:
        return None, None, None, None, f"Missing files or error loading {model_name}: {str(e)}"

model, scaler, model_columns, model_type, load_status = load_artifacts(model_choice)

if load_status != "Success":
    st.warning(f"⚠️ **{load_status}**")
    st.stop()


# --- 4. MAIN LAYOUT (LEFT: INPUTS, RIGHT: OUTPUTS) ---
col_left, col_right = st.columns([55, 45], gap="large")

with col_left:
    with st.container(border=True):
        st.markdown("### Patient profile")
        st.caption("Adjust the fields below — the estimate on the right recalculates instantly.")
        st.write("")
        
        # Input Rows
        r1_col1, r1_col2 = st.columns(2)
        age = r1_col1.slider("Age", min_value=1, max_value=100, value=45)
        gender = r1_col2.radio("Gender", ["Male", "Female"], horizontal=True)

        r2_col1, r2_col2 = st.columns(2)
        avg_glucose_level = r2_col1.slider("Avg. glucose level (mg/dL)", min_value=50.0, max_value=280.0, value=92.0)
        bmi = r2_col2.slider("BMI", min_value=10.0, max_value=60.0, value=26.0)

        st.divider()

        r3_col1, r3_col2 = st.columns(2)
        hypertension = r3_col1.radio("Hypertension", ["No", "Yes"], horizontal=True)
        heart_disease = r3_col2.radio("Heart disease", ["No", "Yes"], horizontal=True)

        r4_col1, r4_col2 = st.columns(2)
        ever_married = r4_col1.radio("Ever married", ["Yes", "No"], horizontal=True)
        residence_type = r4_col2.radio("Residence type", ["Urban", "Rural"], horizontal=True)

        st.divider()

        r5_col1, r5_col2 = st.columns(2)
        work_type = r5_col1.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        smoking_status = r5_col2.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# --- 5. PREPROCESS & PREDICT (Live calculation) ---
# Format raw inputs
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

# Create DataFrame & One-Hot Encode
df_patient = pd.DataFrame([input_data])
df_patient = pd.get_dummies(df_patient, columns=['work_type', 'Residence_type', 'smoking_status'])

# Ensure all columns from the chosen model's training set are present
for col in model_columns:
    if col not in df_patient.columns:
        df_patient[col] = 0
df_patient = df_patient[model_columns]

# Scale ONLY if a standalone scaler was loaded (Random Forest / ANN)
if scaler is not None:
    numeric_cols = ['age', 'avg_glucose_level', 'bmi']
    df_patient[numeric_cols] = scaler.transform(df_patient[numeric_cols])

# Predict
# stroke_prob = model.predict_proba(df_patient)[0][1]
# prob_pct = stroke_prob * 100
# --- PREDICT ---

if model_type == "keras":

    # ANN outputs a probability directly
    stroke_prob = float(
        model.predict(
            df_patient,
            verbose=0
        )[0][0]
    )

else:

    # Random Forest / SVM
    stroke_prob = model.predict_proba(
        df_patient
    )[0][1]


prob_pct = stroke_prob * 100

# Determine Risk Tier for Styling
if prob_pct < 15:
    risk_color = "#e8f5e9" # Light green pill bg
    text_color = "#2ea673" # Dark green text
    risk_label = "LOW RISK"
elif prob_pct < 30:
    risk_color = "#fff3e0" # Light yellow bg
    text_color = "#f39c12" # Orange text
    risk_label = "MODERATE RISK"
else:
    risk_color = "#ffebee" # Light red bg
    text_color = "#e74c3c" # Red text
    risk_label = "HIGH RISK"


# --- 6. RIGHT COLUMN OUTPUTS ---
with col_right:
    # 6A. Dark Live Readout Card
    html_card = f"""
    <div style="background-color: #0e2130; padding: 30px; border-radius: 12px; color: white; margin-bottom: 20px;">
        <div style="font-size: 12px; letter-spacing: 2px; color: #6fc2e9; margin-bottom: 25px; font-weight: 600;">LIVE READOUT</div>
        <div style="display: flex; align-items: baseline; gap: 20px;">
            <div style="font-size: 72px; font-weight: bold; font-family: 'Georgia', serif;">{prob_pct:.1f}%</div>
            <div style="background-color: {risk_color}; color: {text_color}; padding: 6px 16px; border-radius: 20px; font-size: 14px; font-weight: bold; letter-spacing: 1px;">{risk_label}</div>
        </div>
    </div>
    """
    st.markdown(html_card, unsafe_allow_html=True)
    
    # 6B. Feature Importance (Static mockup for context)
    with st.container(border=True):
        st.markdown("#### What's driving this")
        st.write("<span style='font-size: 14px; color: #555;'>Top factors the forest weighs most heavily overall (not per-patient), for context.</span>", unsafe_allow_html=True)
        st.write("")
        
        # Custom HTML to create the progress bars like the image
        factors = [
            ("Age", 54.2), ("Glucose level", 11.9), ("BMI", 9.8), 
            ("Work: child/student", 3.8), ("Ever married", 3.3), ("Formerly smoked", 2.7)
        ]
        
        for name, val in factors:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 12px; font-size: 14px;">
                <div style="width: 150px; font-weight: 500;">{name}</div>
                <div style="flex-grow: 1; background-color: #f0f2f6; height: 8px; border-radius: 4px; margin-right: 15px;">
                    <div style="width: {val}%; background-color: #cbd4e1; height: 100%; border-radius: 4px;"></div>
                </div>
                <div style="width: 45px; text-align: right; color: #555;">{val}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 15px 0px;'>", unsafe_allow_html=True)
        st.write("<span style='font-size: 12px; color: #8898aa;'>This tool reflects patterns learned from a single ~5,000-row public dataset (~5% stroke prevalence) and is built for coursework demonstration, not clinical use. On held-out test data the underlying model catches roughly 66% of real stroke cases (recall) but has low precision (~13%) — meaning most positive flags are false alarms. Do not use this for real medical decisions.</span>", unsafe_allow_html=True)

# --- 7. FOOTER METRICS ---
st.write("")
st.write("")

# Dynamic metrics based on which model is selected in the dropdown
model_metrics = {
    "Random Forest": {"acc": "76.3%", "rec": "66.0%", "prec": "12.8%", "auc": "0.806"},
    "Support Vector Machine (SVM)": {"acc": "73.2%", "rec": "78.4%", "prec": "11.5%", "auc": "0.760"}, # Replace with your actual SVM stats
    "Artificial Neural Network (ANN)": {"acc": "82.6%", "rec": "62.0%", "prec": "16.32%", "auc": "0.807"} # Replace with your actual ANN stats
}

curr_metrics = model_metrics[model_choice]

m1, m2, m3, m4 = st.columns(4)
m1.metric("ACCURACY", curr_metrics["acc"])
m2.metric("RECALL (STROKE)", curr_metrics["rec"])
m3.metric("PRECISION (STROKE)", curr_metrics["prec"])
m4.metric("ROC-AUC", curr_metrics["auc"])