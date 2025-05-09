import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)

label_encoder = joblib.load(Path("Models/label_encoder.joblib"))
scaler = joblib.load(Path("Models/Scaler.joblib"))

def load_data(path: str) -> pd.DataFrame:
    try:
        if not path(path).exists():
            raise FileNotFoundError(f"{path} does not exixts")
        
        if path.endswith(".parquet"):
            return pd.read_parquet(path, engine='pyarrow')
        
        elif path.endswith(".csv"):
            return pd.read_csv(path)
        
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def preprocess(data: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    if data.isnull().sum().sum() > 0:
        data = data.dropna()

    for col in data.select_dtypes(include="category").columns:
        data[col] = label_encoder.transform(data[col])

    y = data['label']
    drop_cols = ['attack_cat', 'label']
    X = data.drop(columns=[col for col in drop_cols if col in data.columns])
    
    X_preprocessed = scaler.transform(X)

    return X_preprocessed, y

    