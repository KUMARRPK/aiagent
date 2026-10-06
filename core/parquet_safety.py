import pandas as pd


def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype != object:
            continue
        sample = df[col].dropna()
        if sample.empty:
            continue
        first_val = sample.iloc[0]
        if not isinstance(first_val, (str, int, float, bool)):
            df[col] = df[col].astype(str)
    return df