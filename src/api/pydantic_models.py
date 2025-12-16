# src/api/pydantic_models.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# NOTE: This model must contain all columns present in your raw/cleaned data (X)
# that are fed into the full_pipeline.

class CustomerData(BaseModel):
    CustomerId: int
    TransactionStartTime: datetime  # Used by FeatureCreator
    Value: float
    # These are placeholder columns that your original data likely contains:
    # Adjust these based on your specific dataset's categorical and numerical columns!
    IsDebit: Optional[bool] = True
    IsLoyaltyMember: Optional[bool] = False
    Country: Optional[str] = "Kenya"
    Currency: Optional[str] = "KES"
    ProviderId: Optional[int] = 1

    # Ensures the input data types are correct
    class Config:
        json_schema_extra = {
            "example": {
                "CustomerId": 12345,
                "TransactionStartTime": "2025-12-16T10:00:00",
                "Value": 1500.50,
                "IsDebit": True,
                "IsLoyaltyMember": False,
                "Country": "Kenya",
                "Currency": "KES",
                "ProviderId": 1
            }
        }

class PredictionResponse(BaseModel):
    is_high_risk_probability: float
    model_version: str