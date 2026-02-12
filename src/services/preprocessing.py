from typing import Iterable, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def infer_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[Iterable[str], Iterable[str]]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore")
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    categorical_cols = [c for c in categorical_cols if c != target_col]
    return list(numeric_cols), list(categorical_cols)


def build_preprocessor(numeric_cols: Iterable[str], categorical_cols: Iterable[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_cols)),
            ("cat", categorical_pipeline, list(categorical_cols)),
        ],
        remainder="drop",
    )
