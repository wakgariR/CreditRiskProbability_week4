# processor.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from xverse.transformer import WOE
from category_encoders import TargetEncoder 

from config import *

# --- 1 & 2. Custom Transformer for Aggregate and Extracted Features ---

class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates time-based and customer-level aggregate features.
    Assumes df has been loaded with DATETIME_COL and VALUE_COL.
    """
    def __init__(self, datetime_col=DATETIME_COL, customer_id=CUSTOMER_ID):
        self.datetime_col = datetime_col
        self.customer_id = customer_id
        self.agg_df = None # To store aggregation results

    def fit(self, X, y=None):
        # Ensure the datetime column is in datetime format
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])

        # --- Aggregate Features (Step 1) ---
        # Calculate all aggregates
        agg_functions = {
            VALUE_COL: ['sum', 'mean', 'std', 'count'],
            'Amount': [lambda x: x[x > 0].sum(), lambda x: x[x < 0].sum()] # Total Debit and Credit
        }
        
        agg_df = X.groupby(self.customer_id).agg(agg_functions).reset_index()
        
        # Flatten and rename columns
        agg_df.columns = [
            self.customer_id,
            'Total_Transaction_Amount', 'Avg_Transaction_Amount', 
            'StdDev_Transaction_Amount', 'Transaction_Count',
            'Total_Debit_Amount', 'Total_Credit_Amount' # Rename debit/credit columns
        ]
        
        # Fill NaN for StdDev (occurs if count=1)
        agg_df['StdDev_Transaction_Amount'] = agg_df['StdDev_Transaction_Amount'].fillna(0)
        
        self.agg_df = agg_df
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.datetime_col] = pd.to_datetime(X_copy[self.datetime_col])
        
        # --- Extract Time Features (Step 2) ---
        X_copy['TransactionHour'] = X_copy[self.datetime_col].dt.hour
        X_copy['TransactionDay'] = X_copy[self.datetime_col].dt.day
        X_copy['TransactionMonth'] = X_copy[self.datetime_col].dt.month
        X_copy['TransactionYear'] = X_copy[self.datetime_col].dt.year
        
        # --- Merge Aggregate Features (Step 1) ---
        # Merge the aggregated features back into the transaction-level data
        X_transformed = X_copy.merge(self.agg_df, on=self.customer_id, how='left')
        
        return X_transformed.drop(columns=[self.datetime_col, VALUE_COL, 'Amount'])


# --- 3, 4, 5, 6. The Complete Sklearn Pipeline ---

def create_feature_pipeline():
    """
    Chains all Sklearn and WoE transformation steps.
    """
    # Identify feature sets for the ColumnTransformer
    ohe_cols = [c for c in CATEGORICAL_COLS if c not in WOE_COLS]
    woe_cols = WOE_COLS
    scaling_cols = NUMERICAL_COLS + TIME_FEATURES

    # 4 & 5. Numerical Pipeline (Imputation and Standardization)
    numerical_pipeline = Pipeline(steps=[
        # Step 4: Imputation (filling NaNs with the mean)
        ('imputer', SimpleImputer(strategy='mean')),
        # Step 5: Standardization (scaling to mean=0, std=1)
        ('scaler', StandardScaler())
    ])

    # 3. Categorical Pipeline (One-Hot Encoding)
    # Using SimpleImputer for 'most_frequent' just in case a category is missing
    ohe_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 6. WoE Transformation (Requires y for fitting)
    # WoE will be handled separately in the ColumnTransformer after the custom transformers.
    # Note: The WoE library needs to be fitted with the target (y), so it must be 
    # included in a ColumnTransformer that passes y.

    # Combine all steps into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, scaling_cols),
            ('ohe', ohe_pipeline, ohe_cols),
            # WoE is often handled outside the main ColumnTransformer 
            # if the WoE package doesn't fit neatly, or custom logic is used.
            # For simplicity, we assume we use a library that integrates well,
            # or we apply WoE before this stage. Since WoE is typically 
            # the last step before the model, we can treat the WoE columns 
            # as if they will be numerical after WoE transformation.
            # Assuming WoE is applied outside this pipeline for now due to complexity
            # of WoE being target-dependent. 
            ('woe_passthrough', 'passthrough', woe_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor

# --- Main Feature Engineering Pipeline ---

def create_full_pipeline(numerical_cols=NUMERICAL_COLS, 
                         time_features=TIME_FEATURES, 
                         categorical_cols=CATEGORICAL_COLS, 
                         woe_cols=WOE_COLS):
    
    scaling_cols = numerical_cols + time_features
    ohe_cols = [c for c in categorical_cols if c not in woe_cols]
    
    # 1. Define the WoE replacement pipeline (TargetEncoder)
    # TargetEncoder converts categorical values to a float based on the target mean.
    # It requires the target (y) during fit, just like WOE, and it's stable.
    woe_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        # Replacing xverse.WOE logic here
        ('target_encoder', TargetEncoder(min_samples_leaf=20, smoothing=10)) 
    ])
    
    # 2. Define the standard OHE pipeline (for columns NOT in WOE_COLS)
    ohe_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 3. Define the numerical pipeline
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 4. Combine all steps in ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, scaling_cols),
            ('ohe', ohe_pipeline, ohe_cols),
            # TargetEncoder is now part of the pipeline, replacing the problematic xverse call
            ('woe', woe_pipeline, woe_cols) 
        ],
        remainder='drop',
        verbose_feature_names_out=False
    ).set_output(transform='pandas') # Keep output as DataFrame structure
    
    # The full pipeline starts with feature creation, then preprocessing
    full_pipeline = Pipeline(steps=[
        ('features', FeatureCreator()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

# Placeholder for WoE application, which must happen before training
def apply_woe(X, y, woe_cols): 
    """
    Applies WoE transformation to specified columns using xverse.
    X is the full DataFrame output by the pipeline.
    woe_cols is a list of the column names (e.g., ['20', '21', '22']).
    
    This version isolates the WOE transformation to avoid KeyErrors.
    """
    X_woe = X.copy()
    
    # 1. Select the subset of columns to be transformed
    X_subset = X_woe[woe_cols]
    
    # 2. Initialize WOE transformer (using the no-arg constructor)
    woe_transformer = WOE() 
    
    # 3. Fit the transformer on the subset
    woe_transformer.fit(X_subset, y) 
    
    # 4. Transform the subset
    # X_transformed_subset is a DataFrame with the same column names as X_subset
    X_transformed_subset = woe_transformer.transform(X_subset)
    
    # 5. Replace the original columns in the full DataFrame (X_woe) 
    # This replacement uses the column names (woe_cols) to map to the correct location.
    # We must use .values here to avoid any index or column name alignment issues 
    # that could arise between the X_woe index and the X_transformed_subset index.
    X_woe[woe_cols] = X_transformed_subset.values 
    
    print(f"Applied WoE transformation to columns at indices: {woe_cols}")
    
    # Save the transformer for production (predict.py)
    return X_woe, woe_transformer