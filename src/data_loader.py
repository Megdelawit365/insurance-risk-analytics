import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path, sep="|")
    df["TransactionMonth"] = pd.to_datetime(
        df["TransactionMonth"], errors="coerce")
    return df
