from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    """Splits data, handles missing values, and standardizes feature scaling.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    if X.isnull().sum().sum() > 0:
        print("[Preprocessing] Imputing missing feature values with median...")
        X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[Preprocessing] Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler


if __name__ == "__main__":
    
    import os
    from data_loader import load_csv

    csv_path = "california_housing_data.csv"
    if os.path.exists(csv_path):
        df = load_csv(csv_path)
        X_tr, X_te, y_tr, y_te, _ = preprocess_pipeline(df, ["MedInc"], "MedHouseVal")
        print("[Preprocessing Test] X_train_scaled shape:", X_tr.shape)