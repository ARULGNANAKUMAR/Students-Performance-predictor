"""
Handles CSV / Excel upload, validation, column mapping, and preview.
"""

import io
import os
import pandas as pd
from werkzeug.utils import secure_filename
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, COLUMN_MAPPING

REQUIRED_DB_COLUMNS = {
    "hours_studied", "attendance", "sleep_hours",
    "previous_score", "target_exam_score",
}

# Reverse: raw feature name → DB column
_raw_to_db = {k.lower(): v for k, v in COLUMN_MAPPING.items()}
# Also allow DB names directly
_raw_to_db.update({v: v for v in COLUMN_MAPPING.values()})


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_upload(file) -> str:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return path


def read_file(path: str) -> pd.DataFrame:
    ext = path.rsplit(".", 1)[1].lower()
    if ext == "csv":
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    elif ext == "xlsx":
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return df


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns from raw dataset names → DB column names.
    Works case-insensitively.
    """
    rename = {}
    for col in df.columns:
        lower = col.strip().lower().replace(" ", "_")
        if lower in _raw_to_db:
            rename[col] = _raw_to_db[lower]
    return df.rename(columns=rename)


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Clean, validate, and return (cleaned_df, list_of_warnings).
    Rows with critical issues are dropped.
    """
    warnings = []
    original_len = len(df)

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Coerce numeric columns
    numeric_cols = [
        "hours_studied", "attendance", "sleep_hours",
        "previous_score", "tutoring_sessions", "physical_activity",
        "target_exam_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Range validation
    range_checks = {
        "attendance":         (0, 100),
        "previous_score":     (0, 100),
        "target_exam_score":  (0, 100),
        "sleep_hours":        (0, 24),
        "hours_studied":      (0, 24),
        "tutoring_sessions":  (0, 100),
        "physical_activity":  (0, 100),
    }
    for col, (lo, hi) in range_checks.items():
        if col in df.columns:
            mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            if mask.any():
                warnings.append(
                    f"{mask.sum()} rows have {col} outside [{lo}, {hi}] — set to NaN."
                )
                df.loc[mask, col] = None

    # Drop rows missing required columns
    for col in REQUIRED_DB_COLUMNS:
        if col not in df.columns:
            df[col] = None

    before = len(df)
    df = df.dropna(subset=["hours_studied", "attendance", "target_exam_score"])
    dropped = before - len(df)
    if dropped:
        warnings.append(f"{dropped} rows dropped due to missing required values.")

    total_dropped = original_len - len(df)
    if total_dropped:
        warnings.append(f"Total skipped: {total_dropped} of {original_len} rows.")

    return df, warnings


def df_to_student_dicts(df: pd.DataFrame, skip_duplicates: bool, existing_ids: set) -> tuple[list[dict], int]:
    """Convert cleaned DataFrame rows → list of student dicts ready for DB insert."""
    records = []
    skipped = 0

    str_cols = ["study_motivation", "internet_usage", "extracurricular",
                "resources_access", "parental_support", "student_identifier"]

    for _, row in df.iterrows():
        sid = str(row.get("student_identifier", "")).strip() or None
        if skip_duplicates and sid and sid in existing_ids:
            skipped += 1
            continue

        record = {
            "student_identifier": sid,
            "hours_studied":      _safe_float(row.get("hours_studied")),
            "attendance":         _safe_float(row.get("attendance")),
            "sleep_hours":        _safe_float(row.get("sleep_hours")),
            "previous_score":     _safe_float(row.get("previous_score")),
            "tutoring_sessions":  _safe_int(row.get("tutoring_sessions")),
            "physical_activity":  _safe_int(row.get("physical_activity")),
            "study_motivation":   _safe_str(row.get("study_motivation")),
            "internet_usage":     _safe_str(row.get("internet_usage")),
            "extracurricular":    _safe_str(row.get("extracurricular")),
            "resources_access":   _safe_str(row.get("resources_access")),
            "parental_support":   _safe_str(row.get("parental_support")),
            "target_exam_score":  _safe_float(row.get("target_exam_score")),
        }
        records.append(record)

    return records, skipped


def preview_dataframe(df: pd.DataFrame, n: int = 10) -> list[dict]:
    return df.head(n).fillna("").to_dict(orient="records")


# ── helpers ──────────────────────────────────────────────────

def _safe_float(v):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(v):
    f = _safe_float(v)
    return None if f is None else int(f)


def _safe_str(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v).strip() or None
