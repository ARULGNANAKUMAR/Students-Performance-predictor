import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ─── Database ───────────────────────────────────────────────
DATABASE_PATH = os.path.join(BASE_DIR, "student_performance.db")
DATABASE_URI = f"sqlite:///{DATABASE_PATH}"

# ─── Model ──────────────────────────────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "student_performance_model.pkl")

# ─── Uploads ────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"csv", "xlsx"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# ─── Flask ──────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# ─── ML Feature Configuration ────────────────────────────────
# Modify these to match your dataset columns
TARGET_COLUMN = "Exam_Score"

FEATURE_COLUMNS = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Motivation_Level",
    "Internet_Access",
    "Extracurricular_Activities",
    "Access_to_Resources",
    "Parental_Involvement",
]

# Map raw CSV/Excel column names → internal DB column names
COLUMN_MAPPING = {
    "Hours_Studied": "hours_studied",
    "Attendance": "attendance",
    "Sleep_Hours": "sleep_hours",
    "Previous_Scores": "previous_score",
    "Tutoring_Sessions": "tutoring_sessions",
    "Physical_Activity": "physical_activity",
    "Motivation_Level": "study_motivation",
    "Internet_Access": "internet_usage",
    "Extracurricular_Activities": "extracurricular",
    "Access_to_Resources": "resources_access",
    "Parental_Involvement": "parental_support",
    "Exam_Score": "target_exam_score",
}

# Reverse mapping: DB column → model feature name
DB_TO_FEATURE = {v: k for k, v in COLUMN_MAPPING.items()}

# ─── Performance Levels ──────────────────────────────────────
PERFORMANCE_LEVELS = [
    (90, 100, "Excellent",         "#10b981"),
    (75,  89, "Very Good",         "#3b82f6"),
    (60,  74, "Good",              "#8b5cf6"),
    (40,  59, "Average",           "#f59e0b"),
    (0,   39, "Needs Improvement", "#ef4444"),
]

def get_performance_level(score):
    for lo, hi, label, color in PERFORMANCE_LEVELS:
        if lo <= score <= hi:
            return label, color
    return "Unknown", "#6b7280"

# ─── Random Forest Hyperparameters ──────────────────────────
RF_PARAMS = {
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1,
}

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Convenience aliases for app.py
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
