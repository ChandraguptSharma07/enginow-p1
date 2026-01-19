# Credit Risk Prediction App

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💰", layout="wide")

@st.cache_resource
def load_model():
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('data/processed/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

st.title("💰 Credit Risk Prediction System")
st.markdown("Predict whether a loan applicant is likely to default")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    person_age = st.slider("Age", 18, 80, 30)
    person_income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=50000, step=5000)
    person_emp_length = st.slider("Years Employed", 0, 40, 5)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

with col2:
    st.subheader("Loan Information")
    loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, value=10000, step=500)
    loan_int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.0, 0.5)

with st.expander("Additional Details"):
    cb_person_default_on_file = st.radio("Previous Default on File?", ["No", "Yes"])
    cb_person_cred_hist_length = st.slider("Credit History Length (years)", 1, 30, 5)

loan_percent_income = loan_amnt / person_income
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
grade_score = grade_map[loan_grade]
prev_default = 1 if cb_person_default_on_file == "Yes" else 0
risk_score = grade_score + prev_default * 2
income_stability = 1 if (person_home_ownership in ['OWN', 'MORTGAGE'] and person_emp_length > 3) else 0

age_groups = {'age_18-25': 0, 'age_26-35': 0, 'age_36-45': 0, 'age_46-55': 0, 'age_55+': 0}
if person_age <= 25: age_groups['age_18-25'] = 1
elif person_age <= 35: age_groups['age_26-35'] = 1
elif person_age <= 45: age_groups['age_36-45'] = 1
elif person_age <= 55: age_groups['age_46-55'] = 1
else: age_groups['age_55+'] = 1

home_ownership = {'home_MORTGAGE': 0, 'home_OTHER': 0, 'home_OWN': 0, 'home_RENT': 0}
home_ownership[f'home_{person_home_ownership}'] = 1

intent_map = {'intent_DEBTCONSOLIDATION': 0, 'intent_EDUCATION': 0, 'intent_HOMEIMPROVEMENT': 0, 'intent_MEDICAL': 0, 'intent_PERSONAL': 0, 'intent_VENTURE': 0}
intent_map[f'intent_{loan_intent}'] = 1

features = {
    'person_age': person_age, 'person_income': person_income, 'person_emp_length': person_emp_length,
    'loan_amnt': loan_amnt, 'loan_int_rate': loan_int_rate, 'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': cb_person_cred_hist_length, 'grade_score': grade_score,
    'prev_default': prev_default, 'risk_score': risk_score, 'income_stability': income_stability,
    **home_ownership, **intent_map, **age_groups
}

input_df = pd.DataFrame([features])
scale_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'grade_score', 'risk_score']
input_df[scale_cols] = scaler.transform(input_df[scale_cols])

st.markdown("---")
if st.button("🔍 Predict Default Risk", type="primary", use_container_width=True):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    st.markdown("---")
    st.subheader("Prediction Result")
    
    col_result1, col_result2 = st.columns(2)
    with col_result1:
        if prediction == 1:
            st.error("⚠️ HIGH RISK - Likely to Default")
        else:
            st.success("✅ LOW RISK - Likely to Repay")
    with col_result2:
        st.metric(label="Default Probability", value=f"{probability[1]*100:.1f}%")
    
    st.markdown("### Risk Assessment")
    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.progress(float(probability[0]), text=f"Repayment: {probability[0]*100:.1f}%")
    with prob_col2:
        st.progress(float(probability[1]), text=f"Default: {probability[1]*100:.1f}%")
    
    st.markdown("### Key Risk Factors")
    factors = []
    if loan_percent_income > 0.4: factors.append("⚠️ High debt-to-income ratio")
    if grade_score >= 4: factors.append("⚠️ Lower loan grade")
    if prev_default == 1: factors.append("⚠️ Previous default on file")
    if loan_int_rate > 15: factors.append("⚠️ High interest rate")
    
    if factors:
        for f in factors: st.write(f)
    else:
        st.write("✅ No major risk factors")

st.markdown("---")
st.caption("Credit Risk Prediction System | Made by Chandragupt Sharma")
