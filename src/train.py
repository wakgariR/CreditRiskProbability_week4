# train.py (Final Code Block)
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import * # Note: apply_woe is removed from import
from processor import create_full_pipeline 

def run_training_pipeline():
    """
    Loads data, performs feature engineering, and trains a Logistic Regression model.
    """
    print("Starting Feature Engineering and Model Training...")
    
    # 1. Load data and split X, y (This part remains the same)
    try:
        df = pd.read_csv(CLEANED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {CLEANED_DATA_PATH} not found. Please ensure the file exists.")
        return

    if TARGET_COL not in df.columns:
        print(f"Error: Target column '{TARGET_COL}' not found in the loaded data.")
        return

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # 2. Apply the Full Pipeline (Now includes TargetEncoder instead of WoE)
    full_pipeline = create_full_pipeline()
    
    # Fit and transform the data. This pipeline now uses the target 'y' internally 
    # for TargetEncoder. The output should be a stable DataFrame (X_model_ready).
    X_model_ready = full_pipeline.fit_transform(X, y) # Pass y to fit for TargetEncoder
    
    # Save the fitted pipeline for use in predict.py
    joblib.dump(full_pipeline, 'feature_pipeline.pkl')
    print(f"Feature engineering pipeline saved to feature_pipeline.pkl")
    
    # 3. Train the Model (Step 7)
    # TargetEncoder output is numeric, so this works directly.
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
    model.fit(X_model_ready, y) 
    
    # 4. Evaluate and Save
    y_pred_proba = model.predict_proba(X_model_ready)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    
    print(f"\nModel Trained Successfully.")
    print(f"AUC on training data: {auc:.4f}")
    
    # Save the trained model
    joblib.dump(model, 'risk_model.pkl')
    print(f"Trained model saved to risk_model.pkl")

if __name__ == '__main__':
    run_training_pipeline()