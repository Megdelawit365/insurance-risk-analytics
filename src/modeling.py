import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier


def split_data(df, target, test_size=0.2):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=42)


def clean_data(df):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
            df[col] = df[col].replace("nan", np.nan)

    return df


def build_preprocessor(X):
    X = X.copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns

    for col in cat_cols:
        X[col] = X[col].astype(str)

    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def evaluate_classification(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }


def train_regression_models(X_train, y_train):
    pre = build_preprocessor(X_train)
    models = {
        "linear": LinearRegression(),
        "rf": RandomForestRegressor(n_estimators=100, random_state=42),
        "xgb": XGBRegressor(objective="reg:squarederror", n_estimators=100)
    }
    pipelines = {}
    for k, m in models.items():
        pipelines[k] = Pipeline([("prep", pre), ("model", m)])
        pipelines[k].fit(X_train, y_train)
    return pipelines


def train_classification_models(X_train, y_train):
    pre = build_preprocessor(X_train)
    models = {
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgb": XGBClassifier(eval_metric="logloss", n_estimators=100)
    }
    pipelines = {}
    for k, m in models.items():
        pipelines[k] = Pipeline([("prep", pre), ("model", m)])
        pipelines[k].fit(X_train, y_train)
    return pipelines
