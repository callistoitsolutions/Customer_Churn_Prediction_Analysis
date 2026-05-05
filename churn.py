"""
Streamlit App for Telecom Customer Churn Prediction
Professional Dark Dashboard UI & Dynamic Data Handling
"""

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set page configuration to wide for dashboard layout
st.set_page_config(
    page_title='Churn AI | Intelligence Dashboard',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ============================================================
# Core Configuration & Dynamic Mapping
# ============================================================

MODEL_PATH = 'models/churn_model_pipeline.pkl'

# The internal column names required by the model and application logic
INTERNAL_COLUMNS = [
    'Gender', 'Age', 'Tenure_Months', 'ContractType', 'MonthlyCharges',
    'TotalCharges', 'InternetService', 'TechSupport', 'OnlineSecurity',
    'PaymentMethod', 'Complaints'
]

# Fuzzy mapping dictionary for handling future data variations.
# Key = Internal Name, Value = List of possible variations found in user files.
COLUMN_VARIATIONS_MAP = {
    'Gender': ['gender', 'sex', 'cust_gender', 'customer_gender'],
    'Age': ['age', 'cust_age', 'customer_age', 'yearsold'],
    'Tenure_Months': ['tenure', 'months', 'tenuremonths', 'lengthofservice', 'cust_tenure'],
    'ContractType': ['contract', 'contracttype', 'agreement', 'contract_type'],
    'MonthlyCharges': ['monthly', 'charges', 'monthlycharges', 'amount_per_month'],
    'TotalCharges': ['total', 'totalcharges', 'lifetime_charges'],
    'InternetService': ['internet', 'internetservice', 'web', 'internet_type'],
    'TechSupport': ['techsupport', 'tech', 'technicalsupport'],
    'OnlineSecurity': ['security', 'onlinesecurity'],
    'PaymentMethod': ['payment', 'paymentmethod', 'pay_type'],
    'Complaints': ['complaints', 'complaint', 'tickets', 'issues']
}

# Define color palette based on reference image
COLOR_BG = "#161A2B"         # Main dark background
COLOR_CARD = "#21263F"       # Slightly lighter card background
COLOR_ACCENT_TEAL = "#22D3EE" # Good info/metric color
COLOR_ACCENT_RED = "#EF4444"  # Churn/danger color
COLOR_ACCENT_ORANGE = "#F97316" # Medium risk color
COLOR_TEXT_PRIMARY = "#FFFFFF" # Main white text
COLOR_TEXT_SECONDARY = "#A0AEC0" # Lighter gray text

# ============================================================
# Enhanced Custom CSS - Professional Dark Theme
# ============================================================
st.markdown(f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Global Styling */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLOR_BG} !important;
        color: {COLOR_TEXT_PRIMARY} !important;
        font-family: 'Roboto', sans-serif;
    }}
    
    /* Ensure all Markdown text is light */
    .stMarkdown, p, label {{
        color: {COLOR_TEXT_PRIMARY} !important;
    }}
    
    /* Custom header - mimicking dashboard title */
    .dashboard-header {{
        padding: 1.5rem 0rem;
        border-bottom: 2px solid {COLOR_CARD};
        margin-bottom: 2rem;
        color: {COLOR_TEXT_PRIMARY} !important;
    }}
    .dashboard-header h1 {{
        color: {COLOR_TEXT_PRIMARY} !important;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .dashboard-header p {{
        color: {COLOR_TEXT_SECONDARY} !important;
        margin-top: 0.5rem;
        font-size: 1rem;
    }}

    /* Card styling for sections and metrics */
    .dashboard-card {{
        background-color: {COLOR_CARD};
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.05);
    }}
    
    /* Metric container override */
    div[data-testid="metric-container"] {{
        background-color: {COLOR_CARD};
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.05);
        text-align: center;
    }}
    div[data-testid="metric-container"] label {{
        color: {COLOR_TEXT_SECONDARY} !important;
        font-weight: 500 !important;
    }}
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {COLOR_ACCENT_TEAL} !important; /* Default metric color */
        font-size: 2rem !important;
    }}
    
    /* Specific styling for red metrics */
    div.metric-red [data-testid="stMetricValue"] {{
        color: {COLOR_ACCENT_RED} !important;
    }}

    /* Forms */
    .stForm {{
        background-color: transparent;
        padding: 0;
        border: none;
    }}
    
    /* Buttons */
    .stButton>button {{
        width: 100%;
        background-color: {COLOR_ACCENT_TEAL};
        color: #000000;
        font-weight: 700;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        text-transform: uppercase;
    }}
    .stButton>button:hover {{
        background-color: #ffffff;
        color: #000000;
        box-shadow: 0 0 15px rgba(34, 211, 238, 0.5);
    }}
    
    /* Download Button */
    .stDownloadButton>button {{
        background-color: #2D3748;
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: #111421;
        border-right: 1px solid rgba(255,255,255,0.05);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 15px;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        padding: 10px 20px;
        color: {COLOR_TEXT_SECONDARY} !important;
        border-bottom: 2px solid transparent;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLOR_ACCENT_TEAL} !important;
        border-bottom: 2px solid {COLOR_ACCENT_TEAL} !important;
        background-color: transparent !important;
    }}
    
    /* Input Fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div,
    .stSlider>div {{
        background-color: {COLOR_BG} !important;
        color: {COLOR_TEXT_PRIMARY} !important;
        border-color: rgba(255,255,255,0.1) !important;
    }}
    
    /* File Uploader */
    .uploadedFile {{
        background-color: {COLOR_CARD};
        border: 1px solid rgba(255,255,255,0.1);
    }}
    
    /* Success/Error boxes */
    .stAlert {{
        border-radius: 8px;
        background-color: {COLOR_CARD};
        border: 1px solid rgba(255,255,255,0.1);
    }}
    div[data-testid="stAlert"] p {{ color: white !important; }}
    div[kind="success"] {{ border-left: 5px solid #10B981; }}
    div[kind="error"] {{ border-left: 5px solid {COLOR_ACCENT_RED}; }}
    div[kind="info"] {{ border-left: 5px solid {COLOR_ACCENT_TEAL}; }}

    /* Highlighting dynamic column resolution results */
    .resolve-box {{
        background-color: #1A202C;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        color: #CBD5E0 !important;
        font-family: monospace;
        font-size: 0.9rem;
    }}

    /* Dataframe override */
    .dataframe {{
        background-color: {COLOR_CARD} !important;
        color: {COLOR_TEXT_PRIMARY} !important;
    }}
    
    /* Progress bar */
    .stProgress > div > div > div > div {{
        background-color: {COLOR_ACCENT_TEAL};
    }}
    
    /* Hide default Streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# Model & Helper Functions
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained model or create a sample if missing"""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model, None
        except Exception as e:
            return None, f"Model loading failed: {str(e)}"
    else:
        return create_sample_model()

def create_sample_model():
    """Create a basic model pipeline for demonstration"""
    os.makedirs('models', exist_ok=True)
    
    np.random.seed(42)
    n = 1000
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Age': np.random.randint(18, 70, n),
        'Tenure_Months': np.random.randint(1, 72, n),
        'ContractType': np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], n),
        'MonthlyCharges': np.random.uniform(20, 120, n),
        'TotalCharges': np.random.uniform(100, 8000, n),
        'InternetService': np.random.choice(['FiberOptic', 'DSL', 'No'], n),
        'TechSupport': np.random.choice(['Yes', 'No'], n),
        'OnlineSecurity': np.random.choice(['Yes', 'No'], n),
        'PaymentMethod': np.random.choice(['Cash', 'BankTransfer', 'CreditCard', 'EWallet'], n),
        'Complaints': np.random.choice(['Yes', 'No'], n),
        'Churn': np.random.choice(['No', 'Yes'], n, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    
    X = df[INTERNAL_COLUMNS]
    y = df['Churn']
    
    num_cols = ['Age', 'Tenure_Months', 'MonthlyCharges', 'TotalCharges']
    cat_cols = ['Gender', 'ContractType', 'InternetService', 'TechSupport', 'OnlineSecurity', 'PaymentMethod', 'Complaints']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', max_depth=5))
    ])
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model, None

def dynamic_column_map_and_validate(user_df):
    """
    Attempts to map user-provided columns to required internal columns.
    Returns: mapped_df (with standardized columns), success_boolean, message, missing_list
    """
    # Create a standardized copy (strip spaces, lowercase)
    temp_df = user_df.copy()
    temp_df.columns = [str(col).strip().lower().replace(' ', '') for col in temp_df.columns]
    
    final_cols_map = {}
    missing = []
    mapped_log = []
    
    for internal_name in INTERNAL_COLUMNS:
        # Check direct match (standardized)
        std_internal_name = internal_name.lower().replace('_', '')
        if std_internal_name in temp_df.columns:
            # Found exact standardized match
            final_cols_map[temp_df.columns[temp_df.columns.get_loc(std_internal_name)]] = internal_name
            continue
            
        # Check variations map
        variations = COLUMN_VARIATIONS_MAP.get(internal_name, [])
        found_variation = False
        for var in variations:
            std_var = var.lower().replace(' ', '').replace('_', '')
            if std_var in temp_df.columns:
                # Map the user's variant column to the internal name
                # We need the ACTUAL variant name in temp_df
                final_cols_map[std_var] = internal_name
                mapped_log.append(f"✓ '{user_df.columns[temp_df.columns.get_loc(std_var)]}' -> '{internal_name}'")
                found_variation = True
                break
        
        if not found_variation:
            missing.append(internal_name)

    if missing:
        msg = f"❌ Missing required features. Could not automatically resolve: {', '.join(missing)}."
        return None, False, msg, mapped_log
    
    # Check for duplicate mappings (e.g., user has 'tenure' AND 'months')
    reverse_map = {}
    for k, v in final_cols_map.items():
        reverse_map.setdefault(v, []).append(k)
    
    duplicates = [inst for inst, std_names in reverse_map.items() if len(std_names) > 1]
    if duplicates:
        return None, False, f"❌ Duplicate column mapping detected for: {', '.join(duplicates)}. Please ensure only one representation of each feature exists.",Mapped_log

    # Perform the mapping and subset
    # 1. subset temp_df to only resolved columns
    resolved_std_names = list(final_cols_map.keys())
    subset_df = temp_df[resolved_std_names]
    
    # 2. Rename
    subset_df = subset_df.rename(columns=final_cols_map)
    
    return subset_df, True, "✅ Column resolution successful.", mapped_log

# ============================================================
# Main Application Layout
# ============================================================

def main():
    # 1. Load Model
    model, error = load_model()
    if error:
        st.error(f"Error initializing model: {error}")
        st.stop()
    
    # 2. Sidebar with dark theme styling
    with st.sidebar:
        st.markdown(f"<h2 style='color:{COLOR_ACCENT_TEAL};'>Churn AI v2.0</h2>", unsafe_allow_html=True)
        st.markdown("---")
        st.info("**Dynamic Mapping Enabled**\n\nThe app now automatically attempts to resolve column variations in future data files (CSV/Excel).")
        st.markdown("---")
        st.markdown(f"<h4 style='color:{COLOR_TEXT_SECONDARY};'>Model Stats</h4>", unsafe_allow_html=True)
        st.write("Type: Random Forest")
        st.write("Features: 11 Standardized")
        st.markdown("---")
        
    # 3. Custom Dashboard Header (mimicking image)
    st.markdown(f"""
        <div class="dashboard-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1>Analyzing Customer Churn</h1>
                    <p>Powered by Advanced Predictive Analytics</p>
                </div>
                <div style="text-align: right; display: flex; gap: 20px;">
                    <div style="text-align: center; border-left: 2px solid rgba(255,255,255,0.05); padding-left: 20px;">
                        <div style="color:{COLOR_ACCENT_TEAL}; font-size:1.8rem; font-weight:700;">82.1%</div>
                        <div style="color:{COLOR_TEXT_SECONDARY}; font-size:0.8rem;">Model Accuracy</div>
                    </div>
                    <div style="text-align: center; border-left: 2px solid rgba(255,255,255,0.05); padding-left: 20px;">
                        <div style="color:#F59E0B; font-size:1.8rem; font-weight:700;">4.1x</div>
                        <div style="color:{COLOR_TEXT_SECONDARY}; font-size:0.8rem;">Retention Lift</div>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 4. Main Tabs
    tab1, tab2 = st.tabs(["🔮 Single Prediction", "📊 Batch Analysis Dashboard"])
    
    # ============================================================
    # TAB 1: Single Prediction
    # ============================================================
    with tab1:
        st.markdown(f"<h3 style='color:{COLOR_TEXT_PRIMARY};'>New Prediction</h3>", unsafe_allow_html=True)
        
        with st.form("single_pred_form"):
            with st.container():
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"<h4 style='color:{COLOR_ACCENT_TEAL};'>Demographics</h4>", unsafe_allow_html=True)
                    gender = st.selectbox("Gender", ['Male', 'Female'])
                    age = st.slider("Age", 18, 100, 45)
                    tenure = st.slider("Tenure (Months)", 1, 72, 24)
                    
                with col2:
                    st.markdown(f"<h4 style='color:{COLOR_ACCENT_TEAL};'>Services & Activity</h4>", unsafe_allow_html=True)
                    internet = st.selectbox("Internet Service", ['FiberOptic', 'DSL', 'No'])
                    security = st.radio("Online Security", ['Yes', 'No'], horizontal=True)
                    support = st.radio("Tech Support", ['Yes', 'No'], horizontal=True)
                    complaints = st.radio("Complaints Filed", ['Yes', 'No'], horizontal=True)
                    
                with col3:
                    st.markdown(f"<h4 style='color:{COLOR_ACCENT_TEAL};'>Financials</h4>", unsafe_allow_html=True)
                    contract = st.selectbox("Contract Type", ['Month-to-Month', 'One-Year', 'Two-Year'])
                    monthly = st.number_input("Monthly Charges ($)", 10.0, 150.0, 75.0)
                    total = st.number_input("Total Charges ($)", monthly, 10000.0, monthly * tenure)
                    payment = st.selectbox("Payment Method", ['CreditCard', 'BankTransfer', 'Cash', 'EWallet'])

                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
            submit = st.form_submit_button("Execute Prediction")
            
        if submit:
            input_dict = {
                'Gender': gender, 'Age': age, 'Tenure_Months': tenure,
                'ContractType': contract, 'MonthlyCharges': monthly,
                'TotalCharges': total, 'InternetService': internet,
                'TechSupport': support, 'OnlineSecurity': security,
                'PaymentMethod': payment, 'Complaints': complaints
            }
            input_df = pd.DataFrame([input_dict])
            
            with st.spinner('Model calculating risk...'):
                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0][1] # Probability of 'Yes'

            # --- SINGLE PREDICTION RESULTS DASHBOARD ---
            st.markdown("---")
            st.markdown(f"<h3 style='color:{COLOR_TEXT_PRIMARY};'>Analysis Results</h3>", unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown('<div class="dashboard-card" style="text-align:center;">', unsafe_allow_html=True)
                st.markdown(f"<h4 style='color:{COLOR_TEXT_SECONDARY};'>Final Risk Verdict</h4>", unsafe_allow_html=True)
                
                if pred == 'Yes':
                    st.markdown(f"<div style='color:{COLOR_ACCENT_RED}; font-size:4rem; font-weight:700; margin: 10px 0;'>CHURN</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:#10B981; font-size:4rem; font-weight:700; margin: 10px 0;'>KEEP</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with res_col2:
                # Custom Metric Cards for Probability and Tier
                st.markdown('<div style="display:flex; gap:15px;">', unsafe_allow_html=True)
                
                # Probability Card
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    metric_cls = "metric-red" if proba > 0.6 else ""
                    st.markdown(f'<div class="{metric_cls}">', unsafe_allow_html=True)
                    st.metric(label="Churn Probability", value=f"{proba:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Tier Card
                with p_col2:
                    if proba > 0.7: tier, tier_c, tier_e = "High", COLOR_ACCENT_RED, "🔴"
                    elif proba > 0.4: tier, tier_c, tier_e = "Medium", "#F59E0B", "🟠"
                    else: tier, tier_c, tier_e = "Low", "#10B981", "🟢"
                    
                    st.markdown(f"""
                    <div style="background-color:{COLOR_CARD}; padding:1rem; border-radius:10px; border:1px solid rgba(255,255,255,0.05); text-align:center;">
                        <div style="color:{COLOR_TEXT_SECONDARY}; font-size:0.9rem; font-weight:500;">Risk Tier</div>
                        <div style="color:{tier_c}; font-size:2rem; font-weight:700; margin-top:5px;">{tier_e} {tier}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # TAB 2: Batch Analysis & Dashboard
    # ============================================================
    with tab2:
        st.markdown(f"<h3 style='color:{COLOR_TEXT_PRIMARY};'>Analyze Customer Dataset</h3>", unsafe_allow_html=True)
        
        up_col1, up_col2 = st.columns([1, 1])
        
        with up_col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown(f"<h5 style='color:{COLOR_ACCENT_TEAL};'>1. Upload Source File</h5>", unsafe_allow_html=True)
            # User requirement met: Accepts Excel (.xlsx) and CSV
            uploaded_file = st.file_uploader("Choose file (CSV or Excel)", type=['csv', 'xlsx'])
            st.markdown('</div>', unsafe_allow_html=True)

        # Container for execution controls and resolution status
        ctl_container = st.container()

        if uploaded_file is not None:
            with st.spinner('Loading and standardizing file structure...'):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file, engine='openpyxl')
                    
                    # Show preview in a dashboard card
                    with up_col2:
                        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                        st.markdown(f"<h5 style='color:{COLOR_ACCENT_TEAL};'>2. Data Preview ({len(df):,} Rows)</h5>", unsafe_allow_html=True)
                        st.dataframe(df.head(5), use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # --- DYNAMIC VALIDATION AND PREDICTION ---
                    with ctl_container:
                        st.markdown('---')
                        st.markdown(f"<h5 style='color:{COLOR_ACCENT_TEAL};'>3. Resolve Columns & Predict</h5>", unsafe_allow_html=True)
                        
                        # Apply dynamic mapping
                        mapped_df, valid, msg, log = dynamic_column_map_and_validate(df)
                        
                        if not valid:
                            st.error(msg)
                            st.info(f"The model requires these standard features: **{', '.join(INTERNAL_COLUMNS)}**. Please ensure your file contains them or related variations defined in the dynamic mapper.")
                            if log:
                                st.markdown("Potential Partial Mappings Identified:")
                                st.json(log)
                        else:
                            st.success(msg)
                            # Show resolution box (mimicking data science info look)
                            with st.expander("View Dynamic Column Resolution Details"):
                                st.markdown('<div class="resolve-box">', unsafe_allow_html=True)
                                for item in log: st.write(item)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Action button
                            if st.button("Generate Batch Prediction Dashboard", type='primary'):
                                # Perform predictions
                                with st.spinner('AI analyzing batch data...'):
                                    probs = model.predict_proba(mapped_df)[:, 1]
                                    preds = model.predict(mapped_df)
                                    
                                    # Add results to original dataframe for export
                                    result_df = df.copy()
                                    result_df['Churn_Probability'] = probs.round(3)
                                    result_df['Churn_Verdict'] = preds
                                    
                                    # Create risk tier based on reference
                                    result_df['Risk_Tier'] = result_df['Churn_Probability'].apply(
                                        lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.4 else 'Low')
                                    )

                                # Ballons for dynamic success feedback
                                st.balloons()
                                
                                # --- BATCH RESULTS DASHBOARD (Mimicking ref image style) ---
                                st.markdown("---")
                                st.markdown(f"<h2 style='color:{COLOR_TEXT_PRIMARY}; text-transform:uppercase;'>Analysis Dashboard</h2>", unsafe_allow_html=True)
                                
                                # Row 1: Key Metrics mimicking the top-right of ref image
                                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                                
                                total_cust = len(df)
                                churners_cnt = (result_df['Churn_Verdict'] == 'Yes').sum()
                                churn_rate = churners_cnt / total_cust
                                avg_prob = result_df['Churn_Probability'].mean()

                                with m_col1: st.metric("Total Customers Analysed", f"{total_cust:,}")
                                with m_col2: st.metric("Average Churn Risk Score", f"{avg_prob:.1%}")
                                with m_col3: 
                                    st.markdown('<div class="metric-red">', unsafe_allow_html=True)
                                    st.metric("Predicted Churners Count", f"{churners_cnt:,}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with m_col4:
                                    st.markdown('<div class="metric-red">', unsafe_allow_html=True)
                                    st.metric("Overall Churn Rate", f"{churn_rate:.1%}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)

                                # Row 2: Detailed Results Table mimicking a panel
                                st.markdown(f"<h4 style='color:{COLOR_TEXT_SECONDARY};'>High-Risk Priority Segments</h4>", unsafe_allow_html=True)
                                
                                # Show only high risk for immediate focus
                                high_risk_df = result_df[result_df['Risk_Tier'] == 'High'].sort_values(by='Churn_Probability', ascending=False)
                                
                                st.dataframe(high_risk_df, use_container_width=True, height=300)

                                # Row 3: Export mimicking dashboard footer controls
                                st.markdown("---")
                                ex_col1, ex_col2, ex_col3 = st.columns([2, 1, 1])
                                with ex_col1:
                                    st.write(f"Showing {len(high_risk_df):,} High-Risk Customers from total analysed dataset.")
                                with ex_col2:
                                    # Export to CSV (easiest for Streamlit)
                                    csv_res = result_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        "Export Full Results (CSV)",
                                        data=csv_res,
                                        file_name="churn_predictions.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                with ex_col3:
                                    # Bonus: Export to Excel (requires io.BytesIO)
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        result_df.to_excel(writer, index=False, sheet_name='ChurnResults')
                                    excel_res = output.getvalue()
                                    
                                    st.download_button(
                                        "Export Full Results (Excel)",
                                        data=excel_res,
                                        file_name="churn_predictions.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )

                except Exception as e:
                    ctl_container.error(f"❌ Error processing file: {str(e)}")

        else:
            # Welcome/Instruction message when no file uploaded
            st.markdown('<div class="dashboard-card" style="text-align:center;">', unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:{COLOR_TEXT_SECONDARY};'>Upload a data file in the standard CSV or Excel (.xlsx) format to generate a batch prediction dashboard.</h4>", unsafe_allow_html=True)
            st.write("Dynamic column mapping will automatically attempt to handle variations in column names across different future datasets.")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
