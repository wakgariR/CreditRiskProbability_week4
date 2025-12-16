# src/api/main.py
import pandas as pd
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import os
import joblib

# Import Pydantic models
from .pydantic_models import CustomerData, PredictionResponse
from config import TARGET_COL # To ensure we drop the target column if present

# --- MLFLOW CONFIGURATION ---
MODEL_NAME = "RFM_Credit_Risk_Proxy_Model"
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
# If running locally, you must ensure the tracking URI is accessible
# If running in Docker, you'd configure a remote/shared URI.

# --- API Setup ---
app = FastAPI(title="Credit Risk Proxy Model API")

# Global variables for the model
model = None
model_version = "Unknown"

def load_model_from_mlflow(model_name: str):
    """Loads the latest model from the MLflow Model Registry."""
    global model, model_version
    
    # Set the tracking URI (necessary for the client to find the registry)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # Get the latest 'Production' or 'Staging' version
        client = MlflowClient()
        latest_version = client.get_latest_versions(name=model_name, stages=["Production", "Staging"])[0]
        
        # Load the model using MLflow's pyfunc load_model utility
        model_uri = f"runs:/{latest_version.run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)
        model_version = f"v{latest_version.version} (Run ID: {latest_version.run_id[:8]})"
        
        print(f"Successfully loaded model: {model_name} {model_version}")

    except Exception as e:
        print(f"Error loading model from MLflow: {e}. Attempting local fallback.")
        # Fallback to a locally saved model (e.g., the last one saved by train.py)
        try:
            model = joblib.load('../models/risk_model.pkl')
            model_version = "Local Fallback"
            print("Successfully loaded local fallback model.")
        except FileNotFoundError:
             raise RuntimeError("Model artifact not found locally or in MLflow.")


@app.on_event("startup")
async def startup_event():
    """Load model when the API starts."""
    load_model_from_mlflow(MODEL_NAME)

@app.get("/", summary="Health Check")
def read_root():
    """Simple health check endpoint."""
    return {"status": "ok", "model_loaded": model_version}

@app.post("/predict", response_model=PredictionResponse, summary="Predict Credit Risk Probability")
def predict_risk(data: CustomerData):
    """
    Accepts customer data and returns the probability of the customer being
    in the high-risk segment (is_high_risk = 1).
    """
    if model is None:
        return PredictionResponse(is_high_risk_probability=0.5, model_version="Not Loaded")

    # 1. Convert Pydantic model to Pandas DataFrame
    input_data = data.model_dump()
    df = pd.DataFrame([input_data])
    
    # 2. Get Prediction Probability
    # The loaded MLflow model is a Pipeline, so predict_proba handles preprocessing internally.
    try:
        # Prediction returns array, we need the probability for the positive class (index 1)
        proba = model.predict_proba(df)[0][1]
        
    except Exception as e:
        # This handles issues if the pipeline fails during transformation (e.g., missing columns)
        print(f"Prediction failed during model execution: {e}")
        return PredictionResponse(is_high_risk_probability=-1.0, model_version=model_version)

    return PredictionResponse(
        is_high_risk_probability=round(proba, 4),
        model_version=model_version
    )