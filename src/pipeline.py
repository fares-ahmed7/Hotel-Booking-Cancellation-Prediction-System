import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ==========================
# Columns definition
# ==========================
CATEGORICAL_COLS = [
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "deposit_type",
    "customer_type",
    "reserved_room_type",
    "assigned_room_type",
    "arrival_date_month",
    "country",
]

NUMERIC_COLS = None  # will be inferred

# ==========================
# Preprocessing Block
# ==========================

def build_preprocessor(X: pd.DataFrame):

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

    numeric_pipline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipline, numeric_cols),
            ("cat", categorical_pipline, categorical_cols),
        ]
    )

    return preprocessor

# ==========================
# Model Builders (Pipeline ready)
# ==========================

def build_pipeline(model, X):

    preprocessor = build_preprocessor(X)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]) 

    return pipeline

# ==========================
# Models
# ==========================

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            eval_metric="auc",
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05
        )
    }