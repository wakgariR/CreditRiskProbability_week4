# train.py (Updated for Task 5: MLflow and Multiple Models)
import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline # Use pipeline for GridSearchCV

from config import * 
from processor import create_full_pipeline 

# --- MLFLOW SETUP ---
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Credit_Risk_RFM_Proxy_Modeling")

def evaluate_model(model, X_test, y_test):
    """Calculates and returns standard classification metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    return metrics

def run_training_pipeline():
    print("Starting Model Training and MLflow Tracking...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(CLEANED_DATA_PATH)
        # Use the correct TARGET_COL defined in config.py ('is_high_risk')
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Data Preparation: Split Data (Setting random_state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")

    # 3. Feature Engineering Pipeline (Trained on X_train only)
    full_pipeline = create_full_pipeline()
    
    # Define models and their hyperparameter grids for tuning
    models_to_train = {
        'LogisticRegression': (
            LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'), 
            {'C': [0.1, 1.0, 10.0]} # Simple C parameter tuning
        ),
        'DecisionTree': (
            DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            {'max_depth': [5, 10, 15], 'min_samples_leaf': [10, 50]}
        ),
    }

    best_auc = -1
    best_model_name = ""

    for model_name, (model_instance, param_grid) in models_to_train.items():
        with mlflow.start_run(run_name=f"Training_{model_name}"):
            print(f"\n--- Training {model_name} ---")
            
            # --- 4. Hyperparameter Tuning (Grid Search) ---
            # Create a pipeline combining feature engineering and the model
            model_pipeline = Pipeline(steps=[
                ('preprocessor', full_pipeline),
                ('classifier', model_instance)
            ])
            
            # Use GridSearchCV for tuning (NOTE: This will be slow!)
            # We prefix the param grid keys with 'classifier__'
            tuned_params = {f'classifier__{k}': v for k, v in param_grid.items()}
            
            grid_search = GridSearchCV(
                model_pipeline, 
                tuned_params, 
                cv=3, 
                scoring='roc_auc', 
                n_jobs=-1, 
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # --- 5. Experiment Tracking with MLflow ---
            
            # Log best hyperparameters
            mlflow.log_params(grid_search.best_params_)
            
            # 6. Model Evaluation (on Test Set)
            metrics = evaluate_model(best_model, X_test, y_test)
            mlflow.log_metrics(metrics)
            print(f"Test Set Metrics: {metrics}")

            # Log model artifact
            mlflow.sklearn.log_model(best_model, "model")
            
            # Check if this is the best model overall
            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                best_model_name = model_name
                # Store the best model locally for registration later
                joblib.dump(best_model, f'../models/best_{model_name.lower()}_model.pkl')
            
            mlflow.end_run()

    print(f"\nTraining Complete. Best model found: {best_model_name} (AUC: {best_auc:.4f})")

    # --- 7. Register Best Model in MLflow (Manual step for now, but logged) ---
    # For registration, you would typically run a separate script or use the MLflow UI.
    # We'll log a note here.
    with mlflow.start_run(run_name="Model_Registration_Candidate"):
        mlflow.log_note(f"Candidate for registration: {best_model_name}")

if __name__ == '__main__':
    run_training_pipeline()