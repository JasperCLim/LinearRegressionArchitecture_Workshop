import os
import sqlite3
import pandas as pd
import requests


def load_csv(file_path: str) -> pd.DataFrame:
    """Loads a CSV file into a Pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at path: {file_path}")
    print(f"[DataLoader] Loading CSV from: {file_path}")
    return pd.read_csv(file_path)


def load_from_api(
    url: str, params: dict = None, headers: dict = None
) -> pd.DataFrame:
    """Fetches JSON data from a REST API endpoint and returns a DataFrame."""
    print(f"[DataLoader] Querying API: {url}")
    headers = headers or {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    payload = response.json()
    if "result" in payload and "records" in payload["result"]:
        records = payload["result"]["records"]
    elif isinstance(payload, list):
        records = payload
    else:
        records = payload.get("data", [])

    return pd.DataFrame(records)


def load_from_sqlite(
    db_path: str, query: str, params: tuple = None
) -> pd.DataFrame:
    """Executes a SQL query against an SQLite database and returns a DataFrame."""
    print(f"[DataLoader] Connecting to SQLite DB: {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(query, conn, params=params)
    return df


if __name__ == "__main__":

    csv_path = "california_housing_data.csv"
    if os.path.exists(csv_path):
        df_sample = load_csv(csv_path)
        print("[DataLoader Test] Shape:", df_sample.shape)
        print(df_sample.head(2))
    else:
        print("[DataLoader Test] CSV sample file not found. Skipping test.")