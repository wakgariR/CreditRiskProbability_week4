import pandas as pd
import pytest
from src.processor import FeatureCreator # Assuming FeatureCreator is in src/processor.py
from src.config import CUSTOMER_ID, DATETIME_COL, VALUE_COL 

# Define a minimal sample DataFrame for testing
@pytest.fixture
def sample_data():
    data = {
        CUSTOMER_ID: [101, 101, 102, 102, 103],
        DATETIME_COL: pd.to_datetime(['2023-10-01', '2023-10-05', '2023-11-10', '2023-11-10', '2023-12-01']),
        VALUE_COL: [100, 200, 50, 150, 300],
        'Other_Col': ['A', 'B', 'C', 'D', 'E']
    }
    return pd.DataFrame(data)

# --- Test 1: Feature Engineering Column Creation ---
def test_feature_creator_output_columns(sample_data):
    """Test that the FeatureCreator adds the expected time-based columns."""
    
    creator = FeatureCreator()
    X_transformed = creator.fit_transform(sample_data)
    
    # The expected columns after FeatureCreator runs
    expected_new_cols = ['TransactionHour', 'TransactionDayOfWeek', 'TimeSinceLastTransaction']
    
    for col in expected_new_cols:
        assert col in X_transformed.columns
    
    # Check that original columns needed for the pipeline are preserved
    assert CUSTOMER_ID in X_transformed.columns
    assert VALUE_COL in X_transformed.columns

# --- Test 2: Time Since Last Transaction Calculation ---
def test_time_since_last_transaction_logic(sample_data):
    """Test the logic of the time since last transaction feature."""
    
    creator = FeatureCreator()
    X_transformed = creator.fit_transform(sample_data)
    
    # For Customer 101:
    # Transaction 1: 2023-10-01 (First transaction -> should be 0 or NaN/handled)
    # Transaction 2: 2023-10-05 (Time since last: 4 days)
    
    # We check the seconds for precision.
    customer_101_df = X_transformed[X_transformed[CUSTOMER_ID] == 101]
    
    # The first transaction's 'TimeSinceLastTransaction' should be 0 (or handled by imputation later)
    # We check the second transaction, where the difference is 4 days * 24 hours * 3600 seconds = 345600 seconds
    
    # The actual calculation in FeatureCreator is usually a fillna(0) for the first transaction
    
    # Check the difference for the second transaction (index 1)
    # The value should be > 0. A simplified test:
    
    # Note: If FeatureCreator sets the first transaction to 0, and the second to the actual diff.
    assert customer_101_df['TimeSinceLastTransaction'].iloc[0] == 0.0 # Assuming fillna(0)
    assert customer_101_df['TimeSinceLastTransaction'].iloc[1] > 345500 # Should be around 4 days
    
    # For Customer 102: Both transactions are simultaneous, so TimeSinceLastTransaction should be 0
    customer_102_df = X_transformed[X_transformed[CUSTOMER_ID] == 102]
    assert customer_102_df['TimeSinceLastTransaction'].iloc[0] == 0.0
    assert customer_102_df['TimeSinceLastTransaction'].iloc[1] == 0.0