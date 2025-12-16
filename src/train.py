# train.py
import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # Added for future use, although not currently tuned
from sklearn.pipeline import Pipeline

# Configuration settings (paths and column names)
from config import * # Feature engineering pipeline creation function
from processor import create_full_pipeline 

# --- MLFLOW SETUP ---
# Use a simple SQLite database for local tracking
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Credit_Risk_RFM_Proxy_Modeling")

def evaluate_model(model, X_test, y_test):
    """Calculates and returns standard classification metrics."""
    # Predict on test data
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    return metrics

def run_training_pipeline():
    print("Starting Model Training and MLflow Tracking...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(CLEANED_DATA_PATH)
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
    except Exception as e:
        print(f"Error loading data or target column '{TARGET_COL}': {e}")
        return

    # 2. Data Preparation: Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- CRITICAL FIX: Reset indices to prevent misalignment in CategoryEncoders during CV ---
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    # -----------------------------------------------------------------------------------------
    
    print(f"Data split and indices reset: Train={len(X_train)}, Test={len(X_test)}")

    # 3. Feature Engineering Pipeline (Created once, used in all models)
    full_pipeline = create_full_pipeline()
    
    # Save the fitted pipeline before tuning starts (using a fit on the whole train set)
    joblib.dump(full_pipeline.fit(X_train, y_train), 'feature_pipeline.pkl')
    print(f"Feature engineering pipeline saved to feature_pipeline.pkl")
    
    # Define models and their hyperparameter grids for tuning
    models_to_train = {
        'LogisticRegression': (
            LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'), 
            # Tuning C: lower C = stronger regularization
            {'classifier__C': [0.1, 1.0, 10.0]} 
        ),
        'DecisionTree': (
            DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            # Tuning max_depth and min_samples_leaf
            {'classifier__max_depth': [5, 10, 15], 'classifier__min_samples_leaf': [10, 50]}
        ),
    }

    best_auc = -1
    best_model_run_id = None
    best_model_name = ""

    for model_name, (model_instance, param_grid) in models_to_train.items():
        with mlflow.start_run(run_name=f"Training_{model_name}") as run:
            print(f"\n--- Training {model_name} (GridSearch) ---")
            
            # Create a pipeline combining feature engineering and the model
            model_pipeline = Pipeline(steps=[
                ('preprocessor', full_pipeline), # Use the fitted full_pipeline here or let GS refit
                ('classifier', model_instance)
            ])
            
            # --- 4. Hyperparameter Tuning (Grid Search) ---
            grid_search = GridSearchCV(
                model_pipeline, 
                param_grid, 
                cv=3, 
                scoring='roc_auc', 
                n_jobs=-1, 
                verbose=1
            )
            
            # Fit the Grid Search on the training data
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # --- 5. Experiment Tracking with MLflow ---
            
            # Log best hyperparameters
            mlflow.log_params(grid_search.best_params_)
            
            # 6. Model Evaluation (on Test Set)
            metrics = evaluate_model(best_model, X_test, y_test)
            mlflow.log_metrics(metrics)
            print(f"Test Set Metrics: {metrics}")

            # Log model artifact (the entire pipeline: preprocessing + classifier)
            mlflow.sklearn.log_model(best_model, "model")
            
            # Check for the best model across all runs
            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                best_model_name = model_name
                best_model_run_id = run.info.run_id
                
            mlflow.end_run()

    print(f"\nTraining Complete. Best model found: {best_model_name} (AUC: {best_auc:.4f})")
    
    # --- 7. Model Registration ---
    if best_model_run_id:
        model_uri = f"runs:/{best_model_run_id}/model"
        # Register the best model identified by its run ID and artifact path
        mlflow.register_model(
            model_uri=model_uri,
            name="RFM_Credit_Risk_Proxy_Model"
        )
        print(f"Best model '{best_model_name}' registered to MLflow Model Registry.")


if __name__ == '__main__':
    run_training_pipeline()