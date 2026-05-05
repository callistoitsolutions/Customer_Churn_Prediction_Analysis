"""
Sample Model Training Script
Create a churn prediction model for the dashboard
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ============================================================
# GENERATE SAMPLE TRAINING DATA
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
    
    # Generate target variable with some logic
    # Higher churn probability for:
    # - Month-to-month contracts
    # - High monthly charges
    # - Low tenure
    # - Complaints
    
    churn_prob = (
        (df['contracttype'] == 'Month-to-Month').astype(int) * 0.3 +
        (df['monthlycharges'] > 100).astype(int) * 0.2 +
        (df['tenure_months'] < 12).astype(int) * 0.25 +
        (df['complaints'] == 'Yes').astype(int) * 0.15 +
        np.random.uniform(0, 0.1, n_samples)
    )
    
    df['Churn'] = (churn_prob > 0.5).apply(lambda x: 'Yes' if x else 'No')
    
    return df

# ============================================================
# TRAIN MODEL
# ============================================================

def train_churn_model():
    """Train and save the churn prediction model"""
    
    print("🔄 Generating sample training data...")
    df = generate_sample_data(1000)
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Training set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Define features
    numeric_features = ['age', 'tenure_months', 'monthlycharges', 'totalcharges']
    categorical_features = ['gender', 'contracttype', 'internetservice', 
                          'techsupport', 'onlinesecurity', 'paymentmethod', 'complaints']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])
    
    # Create model pipeline
    print("🤖 Training Random Forest model...")
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        ))
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"✅ Training Accuracy: {train_score:.2%}")
    print(f"✅ Test Accuracy: {test_score:.2%}")
    
    # Save model
    print("💾 Saving model to 'churn_model.pkl'...")
    joblib.dump(model, 'churn_model.pkl')
    
    print("🎉 Model trained and saved successfully!")
    print("\n📋 Model Details:")
    print(f"   - Features: {len(X.columns)}")
    print(f"   - Algorithm: Random Forest")
    print(f"   - Trees: 100")
    print(f"   - Max Depth: 10")
    
    # Save sample data for testing
    print("\n💾 Saving sample test data...")
    sample_data = X_test.head(20).copy()
    sample_data['Actual_Churn'] = y_test.head(20).values
    sample_data.to_csv('sample_test_data.csv', index=False)
    sample_data.to_excel('sample_test_data.xlsx', index=False)
    
    print("✅ Sample files created:")
    print("   - sample_test_data.csv")
    print("   - sample_test_data.xlsx")
    
    return model

if __name__ == "__main__":
    print("="*60)
    print("CHURN PREDICTION MODEL TRAINING")
    print("="*60)
    print()
    
    model = train_churn_model()
    
    print()
    print("="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n🚀 You can now run the dashboard:")
    print("   streamlit run churn_dashboard_pro.py")
    print()
