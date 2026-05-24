import pandas as pd
import os

print(os.getcwd())

df = pd.read_csv(
    "data/MachineLearningRating_v3.txt",
    sep="|"
)

missing_pct = df.isna().mean() * 100

cols_to_drop = missing_pct[
    missing_pct > 50
].index

df = df.drop(columns=cols_to_drop)

df.to_csv(
    "data/MachineLearningRating_cleaned.csv",
    index=False
)
