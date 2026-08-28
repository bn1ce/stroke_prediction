import streamlit as st
import pandas as pd
import joblib
import os
from tensorflow import keras

# --- 1. PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title="Stroke Risk Prediction Engine", layout="wide")

st.markdown("""
<style>
    /* General spacing and background */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    
    /* Header typography */
    .main-title { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 2.8rem; color: #1E3A8A; font-weight: 800; margin-bottom: 0px; padding-bottom: 0px; }
    .subtitle { font-size: 1.1rem; color: #64748B; margin-top: 5px; margin-bottom: 25px; }
    
    /* Metrics row styling */
    div[data-testid="metric-container"] { background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 15px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    
    /* Custom Card Styling for Outputs */
    .readout-card { background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); padding: 30px; border-radius: 15px; color: white; margin-bottom: 20px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
    .readout-card-mini { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 20px; border-radius: 12px; color: white; margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    
    .risk-badge { padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: bold; letter-spacing: 1.2px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    .section-header { color: #1E3A8A; font-weight: 600; margin-top: 15px; margin-bottom: 10px; font-size: 1.2rem; border-bottom: 2px solid #E2E8F0; padding-bottom: 5px;}
</style>
""", unsafe_allow_html=True)

# --- 2. HEADER & MODEL SELECTION ---
col_head_left, col_head_right = st.columns([6, 4])

with col_head_left:
    st.markdown('<div class="main-title"> Stroke Risk Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real-time machine learning inference based on clinical and demographic data.</div>', unsafe_allow_html=True)

with col_head_right:
    model_list = ["Random Forest", "Support Vector Machine (SVM)", "Artificial Neural Network (ANN)"]
    model_choice = st.selectbox(
        "Select Model Architecture:", 
        model_list + ["Compare All 3 Models"]
    )
    st.caption("⚙️ **Engine Config:** SMOTE Balanced | **Dataset:** UCI Healthcare Stroke")

st.markdown("---")

# --- 3. HELPER FUNCTIONS FOR LOADING & PREDICTING ---
@st.cache_resource
def load_artifacts(model_name):
    try:
        if model_name == "Random Forest":
            model = joblib.load('stroke_rf_model.pkl')
            model_columns = joblib.load('rf_columns.pkl') 
            scaler = joblib.load('rf_scaler.pkl')        
            model_type = "sklearn"
            
        elif model_name == "Support Vector Machine (SVM)":
            model = joblib.load('stroke_svm_model.pkl')
            model_columns = joblib.load('svm_columns.pkl')   
            scaler = joblib.load('svm_scaler.pkl')                                    
            model_type = "sklearn"
            
        elif model_name == "Artificial Neural Network (ANN)":
            model = keras.models.load_model('stroke_ann_smote.keras')
            model_columns = joblib.load('ann_columns.pkl') 
            scaler = joblib.load('ann_scaler.pkl')        
            model_type = "keras"

        return model, scaler, model_columns, model_type, "Success"
    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"

def get_prediction(m_name, inp_data):
    """Loads artifacts and returns the stroke probability percentage for a given model."""
    m, s, m_cols, m_type, status = load_artifacts(m_name)
    if status != "Success":
        return None, status
    
    # Format and Encode
    df_p = pd.DataFrame([inp_data])
    df_p = pd.get_dummies(df_p, columns=['work_type', 'Residence_type', 'smoking_status'])
    
    # Align Columns
    for col in m_cols:
        if col not in df_p.columns:
            df_p[col] = 0
    df_p = df_p[m_cols]
    
    # Scale if necessary
    if s is not None:
        numeric_cols = ['age', 'avg_glucose_level', 'bmi']
        df_p[numeric_cols] = s.transform(df_p[numeric_cols])
        
    # Predict
    if m_type == "keras":
        prob = float(m.predict(df_p, verbose=0)[0][0])
    else:
        prob = m.predict_proba(df_p)[0][1]
        
    return prob * 100, "Success"

def get_risk_styling(prob_pct):
    """Returns color codes based on probability tier."""
    if prob_pct < 15:
        return "#D1FAE5", "#15803D", "LOW RISK"
    elif prob_pct < 30:
        return "#FEF3C7", "#B45309", "MODERATE RISK"
    else:
        return "#FEE2E2", "#B91C1C", "HIGH RISK"


# --- 4. MAIN LAYOUT (LEFT: INPUTS, RIGHT: OUTPUTS) ---
col_left, col_right = st.columns([55, 45], gap="large")

with col_left:
    with st.container(border=True):
        st.markdown("### 📋 Patient Profile")
        st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-bottom: 20px;'>Adjust the clinical parameters below. The model's risk estimation will recalculate instantly.</p>", unsafe_allow_html=True)
        
        # GROUP 1: Demographics
        st.markdown('<div class="section-header">🧑‍🤝‍🧑 Demographics & Background</div>', unsafe_allow_html=True)
        r1_col1, r1_col2, r1_col3 = st.columns([1.2, 1, 1])
        age = r1_col1.slider("Age (Years)", min_value=1, max_value=100, value=45)
        gender = r1_col2.radio("Gender", ["Male", "Female"], horizontal=True)
        ever_married = r1_col3.radio("Ever married?", ["Yes", "No"], horizontal=True)

        # GROUP 2: Clinical Metrics
        st.markdown('<div class="section-header">🩺 Clinical Metrics</div>', unsafe_allow_html=True)
        r2_col1, r2_col2 = st.columns(2)
        avg_glucose_level = r2_col1.slider("Avg. Glucose Level (mg/dL)", 50.0, 280.0, 92.0, help="Fasting blood sugar level. Normal is typically below 100 mg/dL.")
        bmi = r2_col2.slider("BMI (Body Mass Index)", 10.0, 60.0, 26.0, help="Normal BMI range is 18.5 to 24.9.")

        r3_col1, r3_col2 = st.columns(2)
        hypertension = r3_col1.radio("Hypertension (High Blood Pressure)", ["No", "Yes"], horizontal=True)
        heart_disease = r3_col2.radio("Heart Disease History", ["No", "Yes"], horizontal=True)

        # GROUP 3: Lifestyle
        st.markdown('<div class="section-header">🏃 Lifestyle Factors</div>', unsafe_allow_html=True)
        r4_col1, r4_col2, r4_col3 = st.columns(3)
        residence_type = r4_col1.radio("Residence Type", ["Urban", "Rural"], horizontal=True)
        work_type = r4_col2.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        smoking_status = r4_col3.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Format raw inputs into dictionary
input_data = {
    'age': age,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'gender': 0 if gender == "Male" else 1,
    'hypertension': 1 if hypertension == "Yes" else 0,
    'heart_disease': 1 if heart_disease == "Yes" else 0,
    'ever_married': 1 if ever_married == "Yes" else 0,
    'work_type': work_type,
    'Residence_type': residence_type,
    'smoking_status': smoking_status
}

# --- 5. RIGHT COLUMN (LIVE READOUT) ---
with col_right:
    
    # ---------------------------------------------------------
    # SCENARIO A: Compare All 3 Models
    # ---------------------------------------------------------
    if model_choice == "Compare All 3 Models":
        st.markdown("### 📊 Live Model Comparison")
        st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-bottom: 20px;'>Observe how different algorithms evaluate the same clinical profile.</p>", unsafe_allow_html=True)
        
        # Loop through all 3 models and display smaller readout cards
        for m_name in model_list:
            prob_pct, status = get_prediction(m_name, input_data)
            
            if status != "Success":
                st.warning(f"⚠️ {m_name} is unavailable: {status}")
                continue
            
            risk_color, text_color, risk_label = get_risk_styling(prob_pct)
            
            # HTML for compact mini-card
            html_card = f"""
            <div class="readout-card-mini">
                <div style="font-size: 12px; letter-spacing: 1px; color: #94A3B8; margin-bottom: 5px; font-weight: 600; text-transform: uppercase;">{m_name}</div>
                <div style="display: flex; align-items: baseline; gap: 15px;">
                    <div style="font-size: 42px; font-weight: 700; font-family: 'Helvetica Neue', sans-serif;">{prob_pct:.1f}%</div>
                    <div class="risk-badge" style="background-color: {risk_color}; color: {text_color};">{risk_label}</div>
                </div>
            </div>
            """
            st.markdown(html_card, unsafe_allow_html=True)
            
    # ---------------------------------------------------------
    # SCENARIO B: Single Model View
    # ---------------------------------------------------------
    else:
        prob_pct, status = get_prediction(model_choice, input_data)
        
        if status != "Success":
            st.warning(f"⚠️ **{status}**")
            st.stop()
            
        risk_color, text_color, risk_label = get_risk_styling(prob_pct)

        # HTML for large readout card
        html_card = f"""
        <div class="readout-card">
            <div style="font-size: 13px; letter-spacing: 2px; color: #94A3B8; margin-bottom: 15px; font-weight: 600;">LIVE RISK ESTIMATION</div>
            <div style="display: flex; align-items: baseline; gap: 20px;">
                <div style="font-size: 75px; font-weight: 800; font-family: 'Helvetica Neue', sans-serif;">{prob_pct:.1f}%</div>
                <div class="risk-badge" style="background-color: {risk_color}; color: {text_color}; font-size: 15px;">{risk_label}</div>
            </div>
            <div style="margin-top: 10px; font-size: 0.9rem; color: #CBD5E1;">Based on the <b>{model_choice}</b> algorithm.</div>
        </div>
        """
        st.markdown(html_card, unsafe_allow_html=True)
        
        # Feature Importance Block
        with st.container(border=True):
            st.markdown("#### 🔍 What's Driving This?")
            st.write("<span style='font-size: 0.9rem; color: #64748B;'>Top predictive factors weighted by the Random Forest model across the entire dataset (for context).</span>", unsafe_allow_html=True)
            st.write("")
            
            factors = [
                ("Age", 54.2), ("Glucose level", 11.9), ("BMI", 9.8), 
                ("Work: child/student", 3.8), ("Ever married", 3.3), ("Formerly smoked", 2.7)
            ]
            
            for name, val in factors:
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 12px; font-size: 0.95rem;">
                    <div style="width: 150px; font-weight: 500; color: #334155;">{name}</div>
                    <div style="flex-grow: 1; background-color: #E2E8F0; height: 8px; border-radius: 4px; margin-right: 15px;">
                        <div style="width: {val}%; background-color: #3B82F6; height: 100%; border-radius: 4px;"></div>
                    </div>
                    <div style="width: 45px; text-align: right; color: #64748B; font-weight: 500;">{val}%</div>
                </div>
                """, unsafe_allow_html=True)

# --- 6. FOOTER METRICS ---
st.write("")
st.markdown("---")

# Dict containing test metrics for all models
model_metrics = {
    "Random Forest": {"Accuracy": "67.8%", "Recall (Stroke)": "84.0%", "Precision (Stroke)": "11.6%", "ROC-AUC": "0.811"},
    "Support Vector Machine (SVM)": {"Accuracy": "74.7%", "Recall (Stroke)": "72.0%", "Precision (Stroke)": "12.8%", "ROC-AUC": "0.816"}, 
    "Artificial Neural Network (ANN)": {"Accuracy": "76.5%", "Recall (Stroke)": "64.0%", "Precision (Stroke)": "12.6%", "ROC-AUC": "0.780"} 
}

# If "Compare All", render a dataframe table of the metrics
if model_choice == "Compare All 3 Models":
    st.markdown("### 📈 Model Performance Benchmark (Hold-out Test Set)")
    st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-bottom: 15px;'>Comparison of algorithms on unseen patient data. Note: Models were tuned to prioritize <b>Recall</b> to minimize false negatives.</p>", unsafe_allow_html=True)
    df_metrics = pd.DataFrame(model_metrics).T
    st.dataframe(df_metrics, use_container_width=True)

# If single model, render the standard 4 metric blocks
else:
    st.markdown(f"### 📈 Model Performance ({model_choice})")
    curr_metrics = model_metrics[model_choice]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ACCURACY", curr_metrics["Accuracy"])
    m2.metric("RECALL (STROKE)", curr_metrics["Recall (Stroke)"], help="Percentage of actual stroke cases correctly identified.")
    m3.metric("PRECISION (STROKE)", curr_metrics["Precision (Stroke)"], help="Percentage of predicted strokes that were actually strokes.")
    m4.metric("ROC-AUC", curr_metrics["ROC-AUC"])