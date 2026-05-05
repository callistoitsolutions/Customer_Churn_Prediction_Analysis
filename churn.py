"""
Customer Churn Prediction Dashboard
Professional Dark Theme | AI-Powered Analytics
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analytics Pro",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1117 !important;
    color: #e8eaf0 !important;
}
.main .block-container { padding: 1.5rem 2rem; }

section[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #2a2f3e;
}
section[data-testid="stSidebar"] * { color: #e8eaf0 !important; }

div[data-testid="metric-container"] {
    background: #1a1f2e !important;
    border: 1px solid #2a2f3e !important;
    border-top: 3px solid #ff6b35 !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}
div[data-testid="metric-container"] label {
    color: #a0a8c0 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ff6b35 !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #161b27 !important;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #2a2f3e;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #a0a8c0 !important;
    border-radius: 7px !important;
    font-weight: 500;
    padding: 8px 20px;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #ff6b35 !important;
    color: #fff !important;
}

.stButton > button {
    background: linear-gradient(135deg, #ff6b35 0%, #ff4500 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255,107,53,0.4) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #00b894 0%, #00a381 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: #1a1f2e !important;
    color: #e8eaf0 !important;
    border: 1px solid #2a2f3e !important;
    border-radius: 8px !important;
}
.stSelectbox svg { fill: #a0a8c0 !important; }

[data-testid="stFileUploader"] {
    background: #1a1f2e !important;
    border: 2px dashed #2a2f3e !important;
    border-radius: 12px !important;
    padding: 1rem;
}
[data-testid="stFileUploader"] * { color: #e8eaf0 !important; }

.streamlit-expanderHeader {
    background: #1a1f2e !important;
    border: 1px solid #2a2f3e !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}
.streamlit-expanderContent {
    background: #161b27 !important;
    border: 1px solid #2a2f3e !important;
}

[data-testid="stDataFrame"] { background: #1a1f2e !important; border-radius: 10px; }
[data-testid="stDataFrame"] * { color: #e8eaf0 !important; }

.stAlert { background: #1a1f2e !important; border-radius: 8px !important; color: #e8eaf0 !important; }
div[data-baseweb="notification"] { background: #1a1f2e !important; color: #e8eaf0 !important; }

.stRadio > div { background: transparent !important; }
.stRadio label { color: #e8eaf0 !important; }

h1, h2, h3, h4, p, label, span { color: #e8eaf0; }
.stMarkdown { color: #e8eaf0 !important; }
.stMarkdown p { color: #e8eaf0 !important; }

#MainMenu, footer { visibility: hidden; }
::-webkit-scrollbar { width: 6px; background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2a2f3e; border-radius: 3px; }

.stat-card {
    background: #1a1f2e;
    border: 1px solid #2a2f3e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    color: #e8eaf0;
}
.section-header {
    font-size: 0.78rem;
    font-weight: 600;
    color: #a0a8c0 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2a2f3e;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS & COLUMN MAPPING
# ─────────────────────────────────────────────
MODEL_PATH = "churn_model.pkl"

CANONICAL_COLS = [
    'Gender', 'Age', 'Tenure_Months', 'ContractType',
    'MonthlyCharges', 'TotalCharges', 'InternetService',
    'TechSupport', 'OnlineSecurity', 'PaymentMethod', 'Complaints'
]

# Fuzzy alias map: lowercase-no-space key -> canonical name
COL_ALIASES = {
    'gender': 'Gender', 'sex': 'Gender',
    'age': 'Age', 'customer_age': 'Age', 'customerage': 'Age',
    'tenure_months': 'Tenure_Months', 'tenure': 'Tenure_Months',
    'months': 'Tenure_Months', 'tenuremonths': 'Tenure_Months',
    'tenure(months)': 'Tenure_Months',
    'contracttype': 'ContractType', 'contract': 'ContractType',
    'contract_type': 'ContractType',
    'monthlycharges': 'MonthlyCharges', 'monthly_charges': 'MonthlyCharges',
    'monthly': 'MonthlyCharges', 'monthlycharge': 'MonthlyCharges',
    'monthlyfee': 'MonthlyCharges',
    'totalcharges': 'TotalCharges', 'total_charges': 'TotalCharges',
    'total': 'TotalCharges', 'totalcharge': 'TotalCharges',
    'totalfee': 'TotalCharges',
    'internetservice': 'InternetService', 'internet': 'InternetService',
    'internet_service': 'InternetService', 'isp': 'InternetService',
    'techsupport': 'TechSupport', 'tech_support': 'TechSupport',
    'technical_support': 'TechSupport', 'technicalsupport': 'TechSupport',
    'onlinesecurity': 'OnlineSecurity', 'online_security': 'OnlineSecurity',
    'security': 'OnlineSecurity',
    'paymentmethod': 'PaymentMethod', 'payment_method': 'PaymentMethod',
    'payment': 'PaymentMethod', 'paymenttype': 'PaymentMethod',
    'complaints': 'Complaints', 'complaint': 'Complaints',
    'has_complaint': 'Complaints', 'complain': 'Complaints',
    'hascomplaint': 'Complaints',
}

ORANGE = '#ff6b35'
GREEN  = '#00cc88'
RED    = '#ff4d4d'
YELLOW = '#ffad33'
BLUE   = '#4a9eff'

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def normalize_columns(df: pd.DataFrame):
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        if key in COL_ALIASES:
            rename_map[col] = COL_ALIASES[key]
    return df.rename(columns=rename_map), rename_map


def validate_columns(df: pd.DataFrame):
    df_n, rmap = normalize_columns(df.copy())
    missing = [c for c in CANONICAL_COLS if c not in df_n.columns]
    return len(missing) == 0, missing, df_n, rmap


def generate_training_data(n=1500):
    np.random.seed(42)
    df = pd.DataFrame({
        'Gender':          np.random.choice(['Male', 'Female'], n),
        'Age':             np.random.randint(18, 72, n),
        'Tenure_Months':   np.random.randint(1, 72, n),
        'ContractType':    np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], n, p=[0.55, 0.30, 0.15]),
        'MonthlyCharges':  np.random.uniform(20, 130, n),
        'InternetService': np.random.choice(['FiberOptic', 'DSL', 'No'], n),
        'TechSupport':     np.random.choice(['Yes', 'No'], n),
        'OnlineSecurity':  np.random.choice(['Yes', 'No'], n),
        'PaymentMethod':   np.random.choice(['Cash', 'BankTransfer', 'CreditCard', 'EWallet'], n),
        'Complaints':      np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
    })
    df['TotalCharges'] = (df['MonthlyCharges'] * df['Tenure_Months']).round(2)
    prob = (0.08
            + (df['ContractType'] == 'Month-to-Month') * 0.28
            + (df['Tenure_Months'] < 12) * 0.18
            + (df['Complaints'] == 'Yes') * 0.35
            + (df['MonthlyCharges'] > 90) * 0.10)
    df['Churn'] = np.where(np.random.random(n) < prob.clip(0, 0.95), 'Yes', 'No')
    return df


@st.cache_resource(show_spinner=False)
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    df = generate_training_data()
    X, y = df[CANONICAL_COLS], df['Churn']
    num_f = ['Age', 'Tenure_Months', 'MonthlyCharges', 'TotalCharges']
    cat_f = ['Gender', 'ContractType', 'InternetService',
             'TechSupport', 'OnlineSecurity', 'PaymentMethod', 'Complaints']
    pre = ColumnTransformer([
        ('num', StandardScaler(), num_f),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_f)
    ])
    pipe = Pipeline([('pre', pre),
                     ('clf', RandomForestClassifier(n_estimators=150, max_depth=12,
                                                    class_weight='balanced', random_state=42))])
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


def plotly_dark_layout():
    return dict(
        plot_bgcolor='#1a1f2e',
        paper_bgcolor='#1a1f2e',
        font_color='#e8eaf0',
        xaxis=dict(gridcolor='#2a2f3e', linecolor='#2a2f3e', tickfont_color='#a0a8c0'),
        yaxis=dict(gridcolor='#2a2f3e', linecolor='#2a2f3e', tickfont_color='#a0a8c0'),
        title_font_color='#e8eaf0',
        legend_font_color='#e8eaf0',
    )


def risk_label(p):
    if p >= 0.70: return "🔴 High"
    if p >= 0.40: return "🟡 Medium"
    return "🟢 Low"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
        <div style='font-size:2.5rem;'>📡</div>
        <div style='font-size:1.1rem; font-weight:700; color:#ff6b35;'>Churn Analytics Pro</div>
        <div style='font-size:0.75rem; color:#a0a8c0; margin-top:4px;'>AI-Powered Retention Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**🤖 Model Info**")
    st.markdown("""
    <div class='stat-card'>
        <div style='color:#a0a8c0;font-size:0.75rem;text-transform:uppercase;'>Algorithm</div>
        <div style='color:#ff6b35;font-weight:600;'>Random Forest (150 trees)</div>
        <div style='color:#a0a8c0;font-size:0.75rem;text-transform:uppercase;margin-top:6px;'>Features</div>
        <div style='color:#ff6b35;font-weight:600;'>11 Attributes</div>
        <div style='color:#a0a8c0;font-size:0.75rem;text-transform:uppercase;margin-top:6px;'>Training Samples</div>
        <div style='color:#ff6b35;font-weight:600;'>1,500 Records</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**⚠️ Risk Thresholds**")
    st.markdown("""
    <div class='stat-card'>
        <div style='margin-bottom:6px;color:#e8eaf0;'><span style='color:#ff4d4d;'>🔴 High</span>&nbsp;&gt; 70%</div>
        <div style='margin-bottom:6px;color:#e8eaf0;'><span style='color:#ffad33;'>🟡 Medium</span>&nbsp;40–70%</div>
        <div style='color:#e8eaf0;'><span style='color:#00cc88;'>🟢 Low</span>&nbsp;&lt; 40%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📂 Accepted Formats**")
    st.markdown("""
    <div class='stat-card' style='color:#e8eaf0;'>
        📊 <b style='color:#ff6b35;'>.xlsx</b> — Excel Workbook<br>
        📋 <b style='color:#ff6b35;'>.xls</b> — Legacy Excel<br>
        📄 <b style='color:#ff6b35;'>.csv</b> — Comma/Semicolon CSV
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📋 Required Columns**")
    with st.expander("View all 11 columns"):
        for c in CANONICAL_COLS:
            st.markdown(f"<span style='color:#ff6b35;'>✓</span> {c}", unsafe_allow_html=True)
        st.caption("Column names auto-matched — minor spelling variations accepted.")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("🔄 Loading AI model..."):
    model = load_or_train_model()

# ─────────────────────────────────────────────
# HEADER BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1a1f2e 0%,#161b27 100%);
            border:1px solid #2a2f3e; border-left:4px solid #ff6b35;
            border-radius:12px; padding:1.4rem 2rem; margin-bottom:1.5rem;
            display:flex; justify-content:space-between; align-items:center;'>
    <div>
        <div style='font-size:1.6rem;font-weight:700;color:#e8eaf0;'>
            📡 ANALYSING CUSTOMER CHURN
        </div>
        <div style='color:#a0a8c0;font-size:0.88rem;margin-top:4px;'>
            Predict · Analyze · Retain — Powered by Machine Learning
        </div>
    </div>
    <div style='text-align:right;'>
        <div style='font-size:0.72rem;color:#a0a8c0;text-transform:uppercase;'>Status</div>
        <div style='color:#00cc88;font-weight:700;font-size:0.95rem;'>● Model Active</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮  Single Prediction",
    "📊  Batch Analysis  (Excel / CSV)",
    "📚  Guide & Info"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### 👤 Enter Customer Profile")

    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("<div class='section-header'>Demographics</div>", unsafe_allow_html=True)
            gender   = st.selectbox("Gender", ['Male', 'Female'])
            age      = st.slider("Age", 18, 80, 35)
            tenure   = st.slider("Tenure (Months)", 1, 72, 12)

        with c2:
            st.markdown("<div class='section-header'>Services</div>", unsafe_allow_html=True)
            contract = st.selectbox("Contract Type", ['Month-to-Month', 'One-Year', 'Two-Year'])
            internet = st.selectbox("Internet Service", ['FiberOptic', 'DSL', 'No'])
            tech     = st.radio("Tech Support", ['Yes', 'No'], horizontal=True)
            security = st.radio("Online Security", ['Yes', 'No'], horizontal=True)

        with c3:
            st.markdown("<div class='section-header'>Billing</div>", unsafe_allow_html=True)
            monthly    = st.number_input("Monthly Charges ($)", 0.0, 300.0, 65.0, 5.0)
            total      = st.number_input("Total Charges ($)", 0.0, 20000.0,
                                          float(round(monthly * tenure, 2)))
            payment    = st.selectbox("Payment Method",
                                       ['Cash', 'BankTransfer', 'CreditCard', 'EWallet'])
            complaints = st.radio("Complaints Filed", ['Yes', 'No'], horizontal=True)

        submitted = st.form_submit_button("🚀 Predict Churn Risk", use_container_width=True)

    if submitted:
        inp = pd.DataFrame([{
            'Gender': gender, 'Age': age, 'Tenure_Months': tenure,
            'ContractType': contract, 'MonthlyCharges': monthly,
            'TotalCharges': total, 'InternetService': internet,
            'TechSupport': tech, 'OnlineSecurity': security,
            'PaymentMethod': payment, 'Complaints': complaints
        }])

        pred    = model.predict(inp)[0]
        proba   = model.predict_proba(inp)[0]
        classes = list(model.classes_)
        cp      = proba[classes.index('Yes')] if 'Yes' in classes else proba[1]
        risk    = "High" if cp >= 0.70 else "Medium" if cp >= 0.40 else "Low"
        rc      = RED if risk == "High" else YELLOW if risk == "Medium" else GREEN

        st.markdown("---")
        st.markdown("#### 📊 Prediction Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Verdict",        pred)
        m2.metric("📈 Churn Prob",     f"{cp:.1%}")
        m3.metric("⚠️ Risk Level",     risk)
        m4.metric("🛡 Retention Prob", f"{1 - cp:.1%}")

        # Gauge chart
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(cp * 100, 1),
            number={'suffix': '%', 'font': {'size': 44, 'color': ORANGE}},
            title={'text': "Churn Risk Score", 'font': {'color': '#e8eaf0', 'size': 15}},
            gauge={
                'axis': {'range': [0, 100],
                         'tickcolor': '#a0a8c0',
                         'tickfont': {'color': '#a0a8c0'}},
                'bar': {'color': rc, 'thickness': 0.28},
                'bgcolor': '#1a1f2e',
                'borderwidth': 0,
                'steps': [
                    {'range': [0,  40], 'color': '#0d2b1d'},
                    {'range': [40, 70], 'color': '#2b2000'},
                    {'range': [70,100], 'color': '#2b0d0d'},
                ],
                'threshold': {'line': {'color': ORANGE, 'width': 3}, 'value': cp * 100}
            }
        ))
        fig_g.update_layout(**plotly_dark_layout(), height=300,
                            margin=dict(t=50, b=10, l=30, r=30))
        st.plotly_chart(fig_g, use_container_width=True)

        # Recommendation panel
        if pred == 'Yes':
            st.markdown(f"""
            <div style='background:#2b0d0d;border:1px solid #ff4d4d;border-radius:10px;padding:1.2rem 1.5rem;'>
                <div style='color:#ff4d4d;font-size:1.1rem;font-weight:700;'>
                    🚨 High Churn Risk Detected ({cp:.1%})
                </div>
                <div style='color:#e8eaf0;margin-top:0.6rem;font-size:0.9rem;'>
                    This customer shows strong churn signals. Immediate action recommended:
                </div>
                <ul style='color:#e8eaf0;margin-top:0.5rem;font-size:0.88rem;line-height:2;'>
                    <li>🎁 Offer a personalised retention discount immediately</li>
                    <li>📞 Contact the customer within 24 hours</li>
                    <li>🔧 Review service quality and resolve any open complaints</li>
                    <li>📋 Present a longer contract with attractive incentives</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#0d2b1d;border:1px solid #00cc88;border-radius:10px;padding:1.2rem 1.5rem;'>
                <div style='color:#00cc88;font-size:1.1rem;font-weight:700;'>
                    ✅ Low Churn Risk ({cp:.1%})
                </div>
                <div style='color:#e8eaf0;margin-top:0.6rem;font-size:0.9rem;'>
                    Customer appears satisfied. Focus on growth opportunities:
                </div>
                <ul style='color:#e8eaf0;margin-top:0.5rem;font-size:0.88rem;line-height:2;'>
                    <li>🌟 Enrol in loyalty / rewards programme</li>
                    <li>📈 Upsell premium or bundled service tiers</li>
                    <li>🎯 Invite to referral programme</li>
                    <li>📊 Continue standard engagement cadence</li>
                </ul>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 📂 Upload Customer Data File")
    st.markdown(
        "<span style='color:#a0a8c0;font-size:0.88rem;'>"
        "Supports <b style='color:#ff6b35;'>.xlsx</b>, "
        "<b style='color:#ff6b35;'>.xls</b>, and "
        "<b style='color:#ff6b35;'>.csv</b>. "
        "Column names are <b style='color:#ff6b35;'>auto-matched</b> — minor spelling differences handled automatically."
        "</span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop file here or click to browse",
        type=['csv', 'xlsx', 'xls'],
        label_visibility="collapsed"
    )

    if uploaded:
        # Read file
        try:
            fname = uploaded.name.lower()
            if fname.endswith('.csv'):
                try:
                    df_raw = pd.read_csv(uploaded)
                except Exception:
                    uploaded.seek(0)
                    df_raw = pd.read_csv(uploaded, sep=';')
            else:
                df_raw = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")
            st.stop()

        st.success(f"✅ Loaded **{len(df_raw):,}** rows and **{len(df_raw.columns)}** columns from `{uploaded.name}`")

        with st.expander("👀 Preview uploaded data (first 5 rows)"):
            st.dataframe(df_raw.head(5), use_container_width=True)

        # Validate
        ok, missing, df_norm, rmap = validate_columns(df_raw.copy())

        if not ok:
            st.error(f"❌ Missing required columns after auto-mapping: **{', '.join(missing)}**")
            st.markdown(f"""
            <div style='background:#1a1f2e;border:1px solid #2a2f3e;border-radius:8px;padding:1rem 1.2rem;color:#e8eaf0;'>
                <b style='color:#ff6b35;'>Your file columns:</b><br>
                <span style='color:#a0a8c0;'>{" &nbsp;|&nbsp; ".join(df_raw.columns.tolist())}</span><br><br>
                <b style='color:#ff6b35;'>Required columns:</b><br>
                <span style='color:#a0a8c0;'>
                Gender, Age, Tenure_Months, ContractType, MonthlyCharges, TotalCharges,
                InternetService, TechSupport, OnlineSecurity, PaymentMethod, Complaints
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            if rmap:
                st.info("🔄 Auto-mapped: " + " · ".join(f"`{k}` → `{v}`" for k, v in rmap.items()))

            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                with st.spinner("🔮 Analysing customers..."):
                    X_b    = df_norm[CANONICAL_COLS].copy()
                    preds  = model.predict(X_b)
                    probas = model.predict_proba(X_b)
                    cls    = list(model.classes_)
                    cp_arr = probas[:, cls.index('Yes')] if 'Yes' in cls else probas[:, 1]

                    df_out = df_raw.copy()
                    df_out['Churn_Prediction']  = preds
                    df_out['Churn_Probability_%'] = (cp_arr * 100).round(1)
                    df_out['Risk_Level'] = [
                        'High' if p >= 0.70 else 'Medium' if p >= 0.40 else 'Low'
                        for p in cp_arr
                    ]

                st.balloons()
                st.markdown("---")
                st.markdown("#### 📊 Summary Dashboard")

                total    = len(df_out)
                churners = (df_out['Churn_Prediction'] == 'Yes').sum()
                cr       = churners / total * 100
                high_r   = (df_out['Risk_Level'] == 'High').sum()
                med_r    = (df_out['Risk_Level'] == 'Medium').sum()
                low_r    = (df_out['Risk_Level'] == 'Low').sum()

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("👥 Total",          f"{total:,}")
                m2.metric("📈 Churn Rate",      f"{cr:.1f}%")
                m3.metric("⚠️ Churners",        f"{churners:,}")
                m4.metric("🔴 High Risk",       f"{high_r:,}")
                m5.metric("🟡 Medium Risk",     f"{med_r:,}")
                m6.metric("🟢 Low Risk",        f"{low_r:,}")

                st.markdown("---")

                # Charts row 1
                r1c1, r1c2, r1c3 = st.columns(3)

                with r1c1:
                    risk_counts = df_out['Risk_Level'].value_counts()
                    order = [r for r in ['High', 'Medium', 'Low'] if r in risk_counts.index]
                    colors = {'High': RED, 'Medium': YELLOW, 'Low': GREEN}
                    fig_d = go.Figure(go.Pie(
                        labels=[o for o in order],
                        values=[risk_counts[o] for o in order],
                        hole=0.62,
                        marker_colors=[colors[o] for o in order],
                        textfont_color='#e8eaf0',
                        hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
                    ))
                    fig_d.update_layout(**plotly_dark_layout(),
                                        title='Risk Distribution',
                                        height=290,
                                        margin=dict(t=40, b=10, l=10, r=10),
                                        legend=dict(font_color='#e8eaf0'))
                    st.plotly_chart(fig_d, use_container_width=True)

                with r1c2:
                    if 'ContractType' in df_norm.columns:
                        ct = df_norm.copy()
                        ct['Churned'] = (preds == 'Yes').astype(int)
                        ct_grp = ct.groupby('ContractType')['Churned'].mean().mul(100).reset_index()
                        ct_grp.columns = ['Contract', 'Churn%']
                        fig_b = px.bar(ct_grp, x='Contract', y='Churn%',
                                       color='Churn%',
                                       color_continuous_scale=[[0, GREEN], [0.5, YELLOW], [1, RED]],
                                       title='Churn % by Contract',
                                       text_auto='.1f')
                        fig_b.update_layout(**plotly_dark_layout(), height=290,
                                            margin=dict(t=40, b=10, l=10, r=10),
                                            coloraxis_showscale=False)
                        fig_b.update_traces(textfont_color='#e8eaf0',
                                            textposition='outside')
                        st.plotly_chart(fig_b, use_container_width=True)

                with r1c3:
                    fig_h = px.histogram(
                        df_out, x='Churn_Probability_%',
                        nbins=20,
                        title='Churn Probability Distribution',
                        color_discrete_sequence=[ORANGE]
                    )
                    fig_h.update_layout(**plotly_dark_layout(), height=290,
                                        margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fig_h, use_container_width=True)

                # Charts row 2
                r2c1, r2c2 = st.columns(2)

                with r2c1:
                    if 'Tenure_Months' in df_norm.columns:
                        bands = pd.cut(df_norm['Tenure_Months'],
                                       bins=[0, 6, 12, 24, 36, 72],
                                       labels=['< 6 months', '6–12 months',
                                               '1–2 years', '2–3 years', '3+ years'])
                        ten_df = pd.DataFrame({'Band': bands, 'Churned': (preds == 'Yes')})
                        ten_grp = ten_df.groupby('Band', observed=True)['Churned'].sum().reset_index()
                        ten_grp.columns = ['Tenure', 'Churners']
                        fig_t = px.bar(ten_grp, x='Churners', y='Tenure',
                                       orientation='h',
                                       title='Churners by Customer Tenure',
                                       color='Churners',
                                       color_continuous_scale=[[0, GREEN], [0.5, ORANGE], [1, RED]],
                                       text_auto=True)
                        fig_t.update_layout(**plotly_dark_layout(), height=290,
                                            margin=dict(t=40, b=10, l=10, r=10),
                                            coloraxis_showscale=False)
                        fig_t.update_traces(textfont_color='#e8eaf0')
                        st.plotly_chart(fig_t, use_container_width=True)

                with r2c2:
                    if 'MonthlyCharges' in df_norm.columns:
                        mc_df = pd.DataFrame({
                            'Churn': preds,
                            'MonthlyCharges': df_norm['MonthlyCharges'].values
                        })
                        fig_bx = px.box(mc_df, x='Churn', y='MonthlyCharges',
                                        color='Churn',
                                        color_discrete_map={'Yes': RED, 'No': GREEN},
                                        title='Monthly Charges vs Churn')
                        fig_bx.update_layout(**plotly_dark_layout(), height=290,
                                              margin=dict(t=40, b=10, l=10, r=10),
                                              legend_font_color='#e8eaf0')
                        st.plotly_chart(fig_bx, use_container_width=True)

                st.markdown("---")
                st.markdown("#### 📋 Detailed Prediction Results")

                def row_style(row):
                    if row['Risk_Level'] == 'High':
                        c = 'background-color:#2b0d0d; color:#e8eaf0'
                    elif row['Risk_Level'] == 'Medium':
                        c = 'background-color:#2b2000; color:#e8eaf0'
                    else:
                        c = 'background-color:#0d2b1d; color:#e8eaf0'
                    return [c] * len(row)

                st.dataframe(df_out.style.apply(row_style, axis=1),
                             use_container_width=True, height=400)

                st.markdown("---")
                dl1, dl2 = st.columns(2)

                with dl1:
                    csv_bytes = df_out.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download as CSV",
                        csv_bytes,
                        "churn_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )

                with dl2:
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                        df_out.to_excel(writer, index=False, sheet_name='Predictions')
                    st.download_button(
                        "📥 Download as Excel",
                        buf.getvalue(),
                        "churn_predictions.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )


# ══════════════════════════════════════════════════════════════
# TAB 3 — GUIDE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📚 User Guide")
    g1, g2 = st.columns(2)

    with g1:
        st.markdown("""
        <div class='stat-card'>
            <div class='section-header'>🔮 Single Prediction</div>
            <ol style='color:#e8eaf0;padding-left:1.2rem;line-height:2.2;font-size:0.9rem;'>
                <li>Go to <b style='color:#ff6b35;'>Single Prediction</b> tab</li>
                <li>Fill in the customer form</li>
                <li>Click <b style='color:#ff6b35;'>Predict Churn Risk</b></li>
                <li>Review the gauge score and risk level</li>
                <li>Follow recommended retention actions</li>
            </ol>
        </div>
        <div class='stat-card'>
            <div class='section-header'>📊 Understanding Risk Levels</div>
            <div style='color:#e8eaf0;font-size:0.88rem;line-height:2.2;'>
                <span style='color:#ff4d4d;font-weight:600;'>🔴 High (&gt;70%)</span> — Immediate action required<br>
                <span style='color:#ffad33;font-weight:600;'>🟡 Medium (40–70%)</span> — Close monitoring needed<br>
                <span style='color:#00cc88;font-weight:600;'>🟢 Low (&lt;40%)</span> — Standard engagement, upsell
            </div>
        </div>
        """, unsafe_allow_html=True)

    with g2:
        st.markdown("""
        <div class='stat-card'>
            <div class='section-header'>📊 Batch Analysis</div>
            <ol style='color:#e8eaf0;padding-left:1.2rem;line-height:2.2;font-size:0.9rem;'>
                <li>Go to <b style='color:#ff6b35;'>Batch Analysis</b> tab</li>
                <li>Upload your <b style='color:#ff6b35;'>.xlsx / .xls / .csv</b> file</li>
                <li>Preview data and check auto-mapping</li>
                <li>Click <b style='color:#ff6b35;'>Run Batch Prediction</b></li>
                <li>Explore the dashboard charts</li>
                <li>Download results as CSV or Excel</li>
            </ol>
        </div>
        <div class='stat-card'>
            <div class='section-header'>📋 Required Columns (11)</div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:5px;font-size:0.85rem;color:#e8eaf0;'>
                <div><span style='color:#ff6b35;'>✓</span> Gender</div>
                <div><span style='color:#ff6b35;'>✓</span> Age</div>
                <div><span style='color:#ff6b35;'>✓</span> Tenure_Months</div>
                <div><span style='color:#ff6b35;'>✓</span> ContractType</div>
                <div><span style='color:#ff6b35;'>✓</span> MonthlyCharges</div>
                <div><span style='color:#ff6b35;'>✓</span> TotalCharges</div>
                <div><span style='color:#ff6b35;'>✓</span> InternetService</div>
                <div><span style='color:#ff6b35;'>✓</span> TechSupport</div>
                <div><span style='color:#ff6b35;'>✓</span> OnlineSecurity</div>
                <div><span style='color:#ff6b35;'>✓</span> PaymentMethod</div>
                <div><span style='color:#ff6b35;'>✓</span> Complaints</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#1a1f2e;border:1px solid #2a2f3e;border-left:4px solid #ff6b35;
                border-radius:10px;padding:1rem 1.5rem;margin-top:0.5rem;'>
        <span style='color:#ff6b35;font-weight:600;'>💡 Pro Tips:</span>
        <ul style='color:#a0a8c0;margin-top:0.5rem;font-size:0.88rem;line-height:2;'>
            <li>Column names are <b style='color:#e8eaf0;'>auto-matched</b> — e.g. <code>tenure</code>,
                <code>Tenure_Months</code>, <code>months</code> all work</li>
            <li>The model <b style='color:#e8eaf0;'>auto-trains on first launch</b> if no saved model found</li>
            <li>Future uploads just need the same column structure — data values can change freely</li>
            <li>Download results as <b style='color:#e8eaf0;'>Excel or CSV</b> for reporting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
