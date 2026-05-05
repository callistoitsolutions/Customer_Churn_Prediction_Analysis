"""
Customer Churn Prediction - Streamlit Dashboard
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ============================================================
# STEP 1: DEFINE ALL FUNCTIONS FIRST
# ============================================================

def generate_sample_data(n_samples=1000):
    """Generate synthetic churn data for demonstration"""
    np.random.seed(42)

    data = {
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'contracttype': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples),
        'monthlycharges': np.random.uniform(20, 150, n_samples),
        'totalcharges': np.random.uniform(20, 8000, n_samples),
        'internetservice': np.random.choice(['DSL', 'FiberOptic', 'No'], n_samples),
        'techsupport': np.random.choice(['Yes', 'No'], n_samples),
        'onlinesecurity': np.random.choice(['Yes', 'No'], n_samples),
        'paymentmethod': np.random.choice(['ElectronicCheck', 'MailedCheck', 'BankTransfer', 'CreditCard'], n_samples),
        'complaints': np.random.choice(['Yes', 'No'], n_samples)
    }

    df = pd.DataFrame(data)

    churn_prob = (
        (df['contracttype'] == 'Month-to-Month').astype(int) * 0.3 +
        (df['monthlycharges'] > 100).astype(int) * 0.2 +
        (df['tenure_months'] < 12).astype(int) * 0.25 +
        (df['complaints'] == 'Yes').astype(int) * 0.15 +
        np.random.uniform(0, 0.1, n_samples)
    )

    df['Churn'] = (churn_prob > 0.5).apply(lambda x: 'Yes' if x else 'No')
    return df


def train_churn_model():
    """Train and save the churn prediction model"""
    df = generate_sample_data(1000)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = ['age', 'tenure_months', 'monthlycharges', 'totalcharges']
    categorical_features = ['gender', 'contracttype', 'internetservice',
                            'techsupport', 'onlinesecurity', 'paymentmethod', 'complaints']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model


# ============================================================
# STEP 2: TRAIN MODEL IF NOT EXISTS (after functions defined)
# ============================================================

MODEL_PATH = "churn_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.info("🔄 Training model for the first time... please wait.")
    train_churn_model()
    st.success("✅ Model trained and ready!")
    st.rerun()

model = joblib.load(MODEL_PATH)

# ============================================================
# STEP 3: STREAMLIT DASHBOARD
# ============================================================

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("---")

# ── Sidebar Inputs ───────────────────────────────────────────
st.sidebar.header("🔧 Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 70, 35)
tenure_months = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
contracttype = st.sidebar.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
monthlycharges = st.sidebar.slider("Monthly Charges ($)", 20.0, 150.0, 70.0)
totalcharges = st.sidebar.slider("Total Charges ($)", 20.0, 8000.0, 1000.0)
internetservice = st.sidebar.selectbox("Internet Service", ["DSL", "FiberOptic", "No"])
techsupport = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
onlinesecurity = st.sidebar.selectbox("Online Security", ["Yes", "No"])
paymentmethod = st.sidebar.selectbox("Payment Method", ["ElectronicCheck", "MailedCheck", "BankTransfer", "CreditCard"])
complaints = st.sidebar.selectbox("Has Complaints", ["Yes", "No"])

# ── Predict ──────────────────────────────────────────────────
input_data = pd.DataFrame([{
    'gender': gender,
    'age': age,
    'tenure_months': tenure_months,
    'contracttype': contracttype,
    'monthlycharges': monthlycharges,
    'totalcharges': totalcharges,
    'internetservice': internetservice,
    'techsupport': techsupport,
    'onlinesecurity': onlinesecurity,
    'paymentmethod': paymentmethod,
    'complaints': complaints
}])

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0]
churn_prob = probability[list(model.classes_).index('Yes')] * 100

# ── Results ──────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Prediction", "⚠️ Will Churn" if prediction == "Yes" else "✅ Will Stay")

with col2:
    st.metric("Churn Probability", f"{churn_prob:.1f}%")

with col3:
    st.metric("Retention Probability", f"{100 - churn_prob:.1f}%")

st.markdown("---")

# ── Gauge Chart ──────────────────────────────────────────────
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=churn_prob,
    title={'text': "Churn Risk (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "red" if churn_prob > 50 else "green"},
        'steps': [
            {'range': [0, 40], 'color': "#d4edda"},
            {'range': [40, 70], 'color': "#fff3cd"},
            {'range': [70, 100], 'color': "#f8d7da"}
        ]
    }
))
st.plotly_chart(fig_gauge, use_container_width=True)

# ── Sample Data Overview ─────────────────────────────────────
st.subheader("📋 Sample Dataset Overview")
sample_df = generate_sample_data(200)
churn_counts = sample_df['Churn'].value_counts().reset_index()
churn_counts.columns = ['Churn', 'Count']

col4, col5 = st.columns(2)

with col4:
    fig_pie = px.pie(churn_counts, values='Count', names='Churn',
                     title="Churn Distribution", color_discrete_sequence=['#2ecc71', '#e74c3c'])
    st.plotly_chart(fig_pie, use_container_width=True)

with col5:
    fig_bar = px.histogram(sample_df, x='contracttype', color='Churn',
                           title="Churn by Contract Type",
                           color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit | Random Forest Classifier")
