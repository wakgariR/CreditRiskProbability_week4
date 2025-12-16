# --- Data Paths ---
CLEANED_DATA_PATH = '../data/target_engineered_data.csv'
TARGET_DATA_PATH = '../data/target_engineered_data.csv'   
RFM_PATH = '../data/rfm_metrics.csv'
MODEL_PATH = '../models/risk_model.pkl'
PIPELINE_PATH = '../models/feature_pipeline.pkl'

# --- Feature Definitions ---
CUSTOMER_ID = 'AccountId'
DATETIME_COL = 'TransactionStartTime'
TARGET_COL = 'is_high_risk' # This is the target we will define in Deliverable 1
VALUE_COL = 'Value' # The primary column for Monetary features

# --- Feature Groups ---
# Numerical features for scaling/standardization
NUMERICAL_COLS = [
    'Total_Transaction_Amount', 'Avg_Transaction_Amount',
    'Transaction_Count', 'StdDev_Transaction_Amount',
    'Total_Debit_Amount', 'Total_Credit_Amount', # Aggregate features
]

# Categorical features for encoding
CATEGORICAL_COLS = [
    'CurrencyCode', 'CountryCode', 'ProviderId',
    'ProductCategory', 'ChannelId', 'PricingStrategy'
]

# Features to extract time components from DATETIME_COL
TIME_FEATURES = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']

# Final list of features to be used in the model
FINAL_FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS + TIME_FEATURES

# --- WoE/IV Configuration ---
# Columns to apply WoE transformation
WOE_COLS = ['CountryCode', 'ProviderId', 'ProductCategory']
# Note: WoE is typically applied to high-cardinality nominal/ordinal features.

# --- Aggregation Period ---
# If you need to define a look-back window for features (e.g., last 90 days)
# For simplicity here, we aggregate over the entire history.