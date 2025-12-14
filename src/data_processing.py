import pandas as pd
from datetime import timedelta

def convert_transaction_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the TransactionStartTime column to datetime.
    """
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df