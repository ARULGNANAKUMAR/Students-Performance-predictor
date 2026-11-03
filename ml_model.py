"""
Machine-Learning layer:
  - Build preprocessing + Random-Forest pipeline
  - Train / evaluate / save / load
  - Single-record prediction
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    MODEL_PATH, RF_PARAMS, TEST_SIZE, RANDOM_STATE,
    FEATURE_COLUMNS, TARGET_COLUMN, COLUMN_MAPPING, DB_TO_FEATURE,
)

# ── Column sets ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    "Hours_Studied", "Attendance", "Sleep_Hours",
    "Previous_Scores", "Tutoring_Sessions", "Physical_Activity",
]
CATEGORICAL_FEATURES = [
    "Motivation_Level", "Internet_Access",
    "Extracurricular_Activities", "Access_to_Resources",
    "Parental_Involvement",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _build_pipeline():
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, NUMERIC_FEATURES),
        ("cat", cat_transformer, CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(**RF_PARAMS)),
    ])
    return pipeline


def _db_rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Convert list of DB dicts → DataFrame with feature-column names."""
    records = []
    for row in rows:
        r = {}
        for feat in ALL_FEATURES:
            db_col = COLUMN_MAPPING.get(feat, feat.lower())
            r[feat] = row.get(db_col)
        r[TARGET_COLUMN] = row.get("target_exam_score")
        records.append(r)
    return pd.DataFrame(records)


def train_model(rows: list[dict]) -> dict:
    """
    Train a fresh pipeline on *rows* (list of DB dicts).
    Returns evaluation metrics dict.
    Raises ValueError if data is insufficient.
    """
    df = _db_rows_to_dataframe(rows)

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET_COLUMN])
    if len(df) < 10:
        raise ValueError(
            f"Need at least 10 records with a valid exam score to train. "
            f"Got {len(df)}."
        )

    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae  = float(mean_absolute_error(y_test, y_pred))
    mse  = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return {
        "mae":              round(mae,  4),
        "mse":              round(mse,  4),
        "rmse":             round(rmse, 4),
        "r2_score":         round(r2,   4),
        "training_records": len(rows),
        "test_records":     len(X_test),
    }


def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def is_model_trained() -> bool:
    return os.path.exists(MODEL_PATH)


def predict(input_dict: dict) -> float:
    """
    Accepts a dict with either feature names (Hours_Studied …)
    or DB column names (hours_studied …) and returns predicted score.
    """
    pipeline = load_pipeline()
    if pipeline is None:
        raise RuntimeError("Model is not trained yet.")

    # Normalise keys to feature-column names
    normalised = {}
    for feat in ALL_FEATURES:
        db_col = COLUMN_MAPPING.get(feat, feat.lower())
        if feat in input_dict:
            normalised[feat] = input_dict[feat]
        elif db_col in input_dict:
            normalised[feat] = input_dict[db_col]
        else:
            normalised[feat] = None

    df = pd.DataFrame([normalised])
    pred = pipeline.predict(df)[0]
    return float(np.clip(pred, 0, 100))


def get_feature_importance() -> list[dict]:
    """Return feature importances as list of {feature, importance}."""
    pipeline = load_pipeline()
    if pipeline is None:
        return []
    rf = pipeline.named_steps["model"]
    importances = rf.feature_importances_
    # Feature names after ColumnTransformer: numerics first, then cats
    names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    result = [
        {"feature": n, "importance": round(float(i), 4)}
        for n, i in zip(names, importances)
    ]
    return sorted(result, key=lambda x: x["importance"], reverse=True)
