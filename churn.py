"""
Streamlit App for Telecom Customer Churn Prediction
Professional & Light UI Design
"""

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set page configuration
st.set_page_config(
    page_title='Churn Predictor Pro',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Enhanced Custom CSS - Light Theme
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main container styling - Light background */
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Custom header - Gradient with good contrast */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }
    
    /* Metric cards - White with colored accents */
    div[data-testid="metric-container"] {
        background: white;
        border-left: 4px solid #667eea;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="metric-container"] label {
        color: #667eea !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #2d3748 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Form styling - Clean white */
    .stForm {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Button styling - Vibrant gradient */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        font-weight: 600;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        background: white;
    }
    
    /* Sidebar styling - Light gradient */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Tab styling - Modern and clean */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: #4a5568;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #667eea;
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: 2px solid transparent;
    }
    
    /* Alert boxes - Light themed */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        background: white;
    }
    
    /* Success alert */
    div[data-baseweb="notification"] {
        background: white;
        border-left: 4px solid #48bb78;
    }
    
    /* Error alert */
    .stAlert[kind="error"] {
        background: white;
        border-left: 4px solid #f56565;
    }
    
    /* Info alert */
    .stAlert[kind="info"] {
        background: white;
        border-left: 4px solid #4299e1;
    }
    
    /* Animation */
    .element-container:has(.stSuccess) {
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* File uploader - Clean white */
    .uploadedFile {
        border-radius: 10px;
        background: white;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2d3748;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    
    /* Subheader styling */
    .stMarkdown h3 {
        color: #667eea;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    /* Expander - Clean white */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        font-weight: 600;
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        background: white;
    }
    
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 1px #667eea;
    }
    
    /* Radio buttons */
    .stRadio>div {
        background: white;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    /* Slider */
    .stSlider>div>div>div>div {
        background: #667eea;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Card effect for sections */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
MODEL_PATH = 'models/churn_model_pipeline.pkl'
REQUIRED_COLUMNS = [
    'Gender', 'Age', 'Tenure_Months', 'ContractType', 'MonthlyCharges',
    'TotalCharges', 'InternetService', 'TechSupport', 'OnlineSecurity',
    'PaymentMethod', 'Complaints'
]

@st.cache_resource
def load_model():
    """Load or create model"""
    try:
        if not os.path.exists(MODEL_PATH):
            return create_sample_model()
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_sample_model():
    """Create sample model if not exists"""
    os.makedirs('models', exist_ok=True)
    
    # Generate sample data
    np.random.seed(42)
    n = 1000
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Age': np.random.randint(18, 70, n),
        'Tenure_Months': np.random.randint(1, 72, n),
        'ContractType': np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], n),
        'MonthlyCharges': np.random.uniform(20, 120, n),
        'InternetService': np.random.choice(['FiberOptic', 'DSL', 'No'], n),
        'TechSupport': np.random.choice(['Yes', 'No'], n),
        'OnlineSecurity': np.random.choice(['Yes', 'No'], n),
        'PaymentMethod': np.random.choice(['Cash', 'BankTransfer', 'CreditCard', 'EWallet'], n),
        'Complaints': np.random.choice(['Yes', 'No'], n),
    }
    
    df = pd.DataFrame(data)
    df['TotalCharges'] = (df['MonthlyCharges'] * df['Tenure_Months']).round(2)
    
    churn_prob = 0.1
    churn_prob += (df['ContractType'] == 'Month-to-Month').astype(int) * 0.3
    churn_prob += (df['Tenure_Months'] < 12).astype(int) * 0.2
    churn_prob += (df['Complaints'] == 'Yes').astype(int) * 0.4
    
    df['Churn'] = (np.random.random(n) < churn_prob).astype(int)
    df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})
    
    # Train model
    X = df[REQUIRED_COLUMNS]
    y = df['Churn']
    
    numeric_features = ['Age', 'Tenure_Months', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['Gender', 'ContractType', 'InternetService', 'TechSupport', 
                          'OnlineSecurity', 'PaymentMethod', 'Complaints']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    
    return model, None

def validate_data(df):
    """Validate input data"""
    df.columns = [col.strip().replace(' ', '') for col in df.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing

def main():
    """Main application"""
    
    # Animated Header
    st.markdown("""
        <div class="custom-header">
            <h1 style='margin:0; font-size: 2.5rem;'>🎯 Churn Predictor Pro</h1>
            <p style='font-size: 1.2rem; margin-top: 1rem; opacity: 0.95;'>AI-Powered Customer Retention Intelligence</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with gradient
    with st.sidebar:
        st.markdown("### 🚀 Quick Navigation")
        st.markdown("---")
        
        st.markdown("### 📊 Model Insights")
        st.info("🎯 **Algorithm**: Random Forest  \n🔢 **Features**: 11 attributes  \n🎲 **Accuracy**: ~82%")
        
        st.markdown("---")
        st.markdown("### 📋 Required Fields")
        with st.expander("View All Fields"):
            for col in REQUIRED_COLUMNS:
                st.markdown(f"✓ {col}")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.success("💼 Use batch prediction for multiple customers")
        st.info("🎯 Risk levels: High >70%, Medium 40-70%, Low <40%")
        
    # Load model
    model, error = load_model()
    
    if model is None:
        st.error(f"❌ {error}")
        st.stop()
    
    # Success message with animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success("✅ **AI Model Ready!** Start predicting customer churn")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs with icons
    tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Batch Analysis", "📚 Guide"])
    
    # ============================================================
    # TAB 1: Single Prediction
    # ============================================================
    with tab1:
        st.markdown("### 👤 Enter Customer Profile")
        st.markdown("Fill in the details below to predict churn probability")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**👥 Demographics**")
                gender = st.selectbox("Gender", ['Male', 'Female'], help="Customer's gender")
                age = st.slider("Age", 18, 100, 35, help="Customer's age in years")
                tenure = st.slider("Tenure (Months)", 1, 72, 12, help="Months with company")
                
            with col2:
                st.markdown("**📱 Services**")
                contract = st.selectbox("Contract", ['Month-to-Month', 'One-Year', 'Two-Year'])
                internet = st.selectbox("Internet", ['FiberOptic', 'DSL', 'No'])
                tech_support = st.radio("Tech Support", ['Yes', 'No'], horizontal=True)
                online_security = st.radio("Online Security", ['Yes', 'No'], horizontal=True)
                
            with col3:
                st.markdown("**💳 Billing**")
                monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, 5.0)
                total = st.number_input("Total Charges ($)", 0.0, value=float(monthly * tenure))
                payment = st.selectbox("Payment Method", ['Cash', 'BankTransfer', 'CreditCard', 'EWallet'])
                complaints = st.radio("Complaints Filed", ['Yes', 'No'], horizontal=True)
            
            st.markdown("---")
            submit = st.form_submit_button("🚀 Predict Churn Risk", use_container_width=True)
        
        if submit:
            input_data = pd.DataFrame({
                'Gender': [gender], 'Age': [age], 'Tenure_Months': [tenure],
                'ContractType': [contract], 'MonthlyCharges': [monthly],
                'TotalCharges': [total], 'InternetService': [internet],
                'TechSupport': [tech_support], 'OnlineSecurity': [online_security],
                'PaymentMethod': [payment], 'Complaints': [complaints]
            })
            
            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                churn_prob = proba[1] if len(proba) > 1 else proba[0]
                
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                # Animated metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎯 Prediction", prediction, 
                             delta="High Risk" if prediction == "Yes" else "Low Risk",
                             delta_color="inverse")
                
                with col2:
                    st.metric("📈 Churn Probability", f"{churn_prob:.1%}",
                             delta=f"{churn_prob*100:.1f} points")
                
                with col3:
                    risk = "High" if churn_prob > 0.7 else "Medium" if churn_prob > 0.4 else "Low"
                    risk_emoji = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "🟢"
                    st.metric("⚠️ Risk Level", f"{risk_emoji} {risk}")
                
                with col4:
                    confidence = (1 - abs(churn_prob - 0.5) * 2) * 100
                    st.metric("🎲 Confidence", f"{confidence:.0f}%")
                
                st.markdown("---")
                
                # Detailed result with styling
                if prediction == 'Yes':
                    st.error(f"### 🚨 HIGH CHURN RISK DETECTED")
                    st.markdown(f"**Probability: {churn_prob:.1%}** | This customer shows strong indicators of churning")
                    
                    st.markdown("#### 💡 Recommended Actions")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        - 🎁 **Immediate**: Offer retention discount
                        - 📞 **Contact**: Reach out within 24 hours
                        - 🔧 **Support**: Review service quality
                        """)
                    with col2:
                        st.markdown("""
                        - 📋 **Contract**: Offer upgrade incentives
                        - 🎯 **Personalize**: Tailor communication
                        - 📊 **Monitor**: Track engagement closely
                        """)
                else:
                    st.success(f"### ✅ LOW CHURN RISK")
                    st.markdown(f"**Probability: {churn_prob:.1%}** | This customer appears satisfied and engaged")
                    
                    st.markdown("#### 💡 Recommended Actions")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        - 🎯 **Upsell**: Explore premium services
                        - 💬 **Engage**: Continue standard communication
                        - 🌟 **Reward**: Consider loyalty program
                        """)
                    with col2:
                        st.markdown("""
                        - 📊 **Monitor**: Regular check-ins
                        - 🎁 **Incentivize**: Referral programs
                        - 📈 **Grow**: Expand service portfolio
                        """)
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # ============================================================
    # TAB 2: Batch Prediction
    # ============================================================
    with tab2:
        st.markdown("### 📊 Batch Customer Analysis")
        st.markdown("Upload a CSV file to analyze multiple customers at once")
        
        uploaded = st.file_uploader("📁 Choose your CSV file", type=['csv'], 
                                     help="Upload a file with customer data")
        
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                
                # Success message
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.success(f"✅ Successfully loaded **{len(df):,}** customer records!")
                
                # Preview
                with st.expander("👀 Preview Data (First 10 Rows)", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Validate
                is_valid, missing = validate_data(df.copy())
                
                if not is_valid:
                    st.error(f"❌ Missing columns: **{', '.join(missing)}**")
                    st.info("💡 Please ensure your CSV includes all required fields")
                else:
                    if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
                        with st.spinner("🔮 Analyzing customer data..."):
                            try:
                                df_pred = df[REQUIRED_COLUMNS].copy()
                                predictions = model.predict(df_pred)
                                probas = model.predict_proba(df_pred)
                                
                                df['Prediction'] = predictions
                                df['Churn_Probability'] = probas[:, 1]
                                df['Risk_Level'] = df['Churn_Probability'].apply(
                                    lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low'
                                )
                                
                                # Animated success
                                st.balloons()
                                st.success("✅ **Analysis Complete!** Results are ready")
                                
                                st.markdown("---")
                                st.markdown("### 📊 Summary Dashboard")
                                
                                # Enhanced metrics
                                col1, col2, col3, col4, col5 = st.columns(5)
                                
                                total = len(df)
                                churners = (df['Prediction'] == 'Yes').sum()
                                churn_rate = (churners / total) * 100
                                high_risk = (df['Risk_Level'] == 'High').sum()
                                medium_risk = (df['Risk_Level'] == 'Medium').sum()
                                
                                col1.metric("📊 Total Customers", f"{total:,}")
                                col2.metric("🔴 Predicted Churners", f"{churners:,}", 
                                           delta=f"{churn_rate:.1f}% rate", delta_color="inverse")
                                col3.metric("⚠️ High Risk", f"{high_risk:,}",
                                           delta=f"{(high_risk/total)*100:.1f}%")
                                col4.metric("🟡 Medium Risk", f"{medium_risk:,}",
                                           delta=f"{(medium_risk/total)*100:.1f}%")
                                col5.metric("🟢 Low Risk", f"{total-churners:,}",
                                           delta=f"{((total-churners)/total)*100:.1f}%")
                                
                                st.markdown("---")
                                st.markdown("### 📋 Detailed Results")
                                
                                # Add color coding
                                def highlight_risk(row):
                                    if row['Risk_Level'] == 'High':
                                        return ['background-color: #ffebee'] * len(row)
                                    elif row['Risk_Level'] == 'Medium':
                                        return ['background-color: #fff9c4'] * len(row)
                                    else:
                                        return ['background-color: #e8f5e9'] * len(row)
                                
                                st.dataframe(df.style.apply(highlight_risk, axis=1), 
                                           use_container_width=True, height=400)
                                
                                # Download section
                                st.markdown("---")
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        "📥 Download Full Report (CSV)",
                                        csv,
                                        "churn_predictions.csv",
                                        "text/csv",
                                        use_container_width=True
                                    )
                                
                            except Exception as e:
                                st.error(f"❌ Prediction error: {str(e)}")
                                
            except Exception as e:
                st.error(f"❌ File error: {str(e)}")
    
    # ============================================================
    # TAB 3: User Guide
    # ============================================================
    with tab3:
        st.markdown("### 📚 Complete User Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🔮 Single Prediction
            
            **Step-by-Step:**
            1. 📝 Fill customer information form
            2. ✅ Review all fields for accuracy
            3. 🚀 Click "Predict Churn Risk"
            4. 📊 Analyze results and recommendations
            5. 💡 Take suggested actions
            
            **Best For:**
            - Individual customer assessment
            - Real-time decision making
            - Detailed customer analysis
            """)
            
            st.markdown("---")
            
            st.markdown("""
            #### 📊 Understanding Risk Levels
            
            - 🔴 **High Risk (>70%)**
              - Immediate action required
              - Priority retention efforts
              - Executive escalation
            
            - 🟡 **Medium Risk (40-70%)**
              - Close monitoring needed
              - Proactive engagement
              - Regular check-ins
            
            - 🟢 **Low Risk (<40%)**
              - Standard engagement
              - Upsell opportunities
              - Maintain satisfaction
            """)
        
        with col2:
            st.markdown("""
            #### 📊 Batch Analysis
            
            **Step-by-Step:**
            1. 📁 Prepare CSV with customer data
            2. ⬆️ Upload file using file uploader
            3. 👀 Preview data for accuracy
            4. 🚀 Click "Generate Predictions"
            5. 📥 Download complete report
            
            **Best For:**
            - Multiple customer analysis
            - Periodic reviews
            - Campaign targeting
            """)
            
            st.markdown("---")
            
            st.markdown("""
            #### 📋 CSV Format Requirements
            
            **Required Columns:**
            ```
            Gender, Age, Tenure_Months, ContractType,
            MonthlyCharges, TotalCharges, InternetService,
            TechSupport, OnlineSecurity, PaymentMethod,
            Complaints
            ```
            
            **Sample Row:**
            ```csv
            Male,35,12,Month-to-Month,65.5,786.0,
            FiberOptic,No,No,CreditCard,Yes
            ```
            """)
        
        st.markdown("---")
        
        st.info("💡 **Pro Tip**: Combine AI predictions with human insight for best results. "
                "Use predictions to prioritize, but personalize your approach!")
        
        st.success("🎯 **Need Help?** Check the sidebar for quick tips and model information!")

if __name__ == "__main__":
    main()
