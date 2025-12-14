import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Assuming config.py is available for column definitions
from config import * # Define paths for saving artifacts
RFM_PATH = '../data/rfm_metrics.csv'
TARGET_DATA_PATH = '../data/target_engineered_data.csv'

def calculate_rfm(df):
    """Calculates Recency, Frequency, and Monetary metrics."""
    print("Calculating RFM metrics...")
    
    # 1. Define Snapshot Date (e.g., one day after the last transaction)
    # Ensure DATETIME_COL is in datetime format first
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    snapshot_date = df[DATETIME_COL].max() + pd.Timedelta(days=1)
    
    # Calculate RFM
    rfm_df = df.groupby(CUSTOMER_ID).agg(
        # Recency: Days since last transaction
        Recency=(DATETIME_COL, lambda x: (snapshot_date - x.max()).days),
        # Frequency: Total number of transactions
        Frequency=(CUSTOMER_ID, 'count'),
        # Monetary: Sum of transaction values
        Monetary=(VALUE_COL, 'sum')
    ).reset_index()
    
    print(f"RFM metrics calculated for {len(rfm_df)} customers.")
    return rfm_df

def cluster_and_label_risk(rfm_df):
    """Scales RFM data, clusters, and assigns the is_high_risk label."""
    
    # --- 2. Pre-process RFM Features ---
    print("Scaling RFM features...")
    rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]
    
    # Handle zero/negative monetary values by adding a small constant or clamping
    # This is a common practice to prepare for log transformation, though we skip log here.
    rfm_features.loc[rfm_features['Monetary'] <= 0, 'Monetary'] = 0.01

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    
    # --- 3. Cluster Customers (K-Means) ---
    K = 3 # Based on instructions
    print(f"Clustering customers into {K} groups...")
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # --- 4. Define and Assign the "High-Risk" Label ---
    
    # Analyze clusters to find the high-risk group (Low Frequency, Low Monetary)
    cluster_analysis = rfm_df.groupby('Cluster').agg(
        Recency_Avg=('Recency', 'mean'),
        Frequency_Avg=('Frequency', 'mean'),
        Monetary_Avg=('Monetary', 'mean'),
        Count=('Cluster', 'count')
    ).sort_values(by=['Frequency_Avg', 'Monetary_Avg'], ascending=[True, True])
    
    print("\n--- Cluster Analysis (Low values indicate High Risk) ---")
    print(cluster_analysis)
    
    # The first cluster in the sorted list is the least engaged (Lowest F and M)
    high_risk_cluster_id = cluster_analysis.index[0]
    
    print(f"\nCluster ID {high_risk_cluster_id} is defined as 'High-Risk' (is_high_risk=1).")
    
    # Assign the binary target variable
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster_id).astype(int)
    
    return rfm_df, scaler, kmeans

def integrate_target_variable(main_df, rfm_df):
    """Merges the new target variable back into the main dataset."""
    print("Merging 'is_high_risk' target into the main dataset...")
    
    # Select only the customer ID and the new target column
    target_map = rfm_df[[CUSTOMER_ID, 'is_high_risk']]
    
    # Merge the target back into the main data based on CustomerId
    # This ensures every transaction is labeled with its customer's risk profile
    final_df = main_df.merge(target_map, on=CUSTOMER_ID, how='left')
    
    # Optional: Drop the original FraudResult column if 'is_high_risk' is the intended proxy.
    # Since you already had FraudResult, we'll keep both for now, but use 'is_high_risk'
    # as the new TARGET_COL for the model if you wish to proceed with the RFM proxy.
    
    print(f"Target integration complete. Final dataset size: {len(final_df)}")
    return final_df

def run_target_engineering():
    # Load the main cleaned transaction data
    try:
        # Assuming the cleaned data is the input source
        df = pd.read_csv(CLEANED_DATA_PATH) 
    except FileNotFoundError:
        print(f"Error: Cannot load {CLEANED_DATA_PATH}. Please run the initial cleaning step.")
        return

    # 1. Calculate RFM
    rfm_metrics = calculate_rfm(df.copy())
    
    # 2. Cluster and Label Risk
    rfm_labeled, rfm_scaler, rfm_kmeans = cluster_and_label_risk(rfm_metrics)
    
    # Save RFM metrics and artifacts
    rfm_labeled.to_csv(RFM_PATH, index=False)
    joblib.dump(rfm_scaler, '../models/rfm_scaler.pkl')
    joblib.dump(rfm_kmeans, '../models/rfm_kmeans.pkl')
    print(f"\nRFM metrics saved to {RFM_PATH}")
    
    # 3. Integrate Target Variable
    final_target_df = integrate_target_variable(df, rfm_labeled)
    
    # Save the final dataset with the new target column
    final_target_df.to_csv(TARGET_DATA_PATH, index=False)
    print(f"Final target-engineered data saved to {TARGET_DATA_PATH}")

if __name__ == '__main__':
    # Ensure config.py is accessible and CLEANED_DATA_PATH is set correctly
    run_target_engineering()