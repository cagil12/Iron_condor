import pandas as pd

try:
    df = pd.read_parquet("wfa_trades.parquet")
    print("Columns:", df.columns.tolist())
    if not df.empty:
        print("First row:", df.iloc[0].to_dict())
except Exception as e:
    print(e)
