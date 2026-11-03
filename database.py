import sqlite3
import json
from datetime import datetime
from config import DATABASE_PATH


def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            student_identifier   TEXT,
            hours_studied        REAL,
            attendance           REAL,
            sleep_hours          REAL,
            previous_score       REAL,
            tutoring_sessions    INTEGER,
            physical_activity    INTEGER,
            study_motivation     TEXT,
            internet_usage       TEXT,
            extracurricular      TEXT,
            resources_access     TEXT,
            parental_support     TEXT,
            target_exam_score    REAL,
            created_at           TEXT DEFAULT (datetime('now')),
            updated_at           TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            student_identifier TEXT,
            input_data         TEXT,
            predicted_score    REAL,
            performance_level  TEXT,
            created_at         TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS model_metrics (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name       TEXT,
            training_records INTEGER,
            mae              REAL,
            mse              REAL,
            rmse             REAL,
            r2_score         REAL,
            trained_at       TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


# ─── Student CRUD ────────────────────────────────────────────

def insert_student(data: dict) -> int:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO students
            (student_identifier, hours_studied, attendance, sleep_hours,
             previous_score, tutoring_sessions, physical_activity,
             study_motivation, internet_usage, extracurricular,
             resources_access, parental_support, target_exam_score)
        VALUES
            (:student_identifier, :hours_studied, :attendance, :sleep_hours,
             :previous_score, :tutoring_sessions, :physical_activity,
             :study_motivation, :internet_usage, :extracurricular,
             :resources_access, :parental_support, :target_exam_score)
    """, data)
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id


def bulk_insert_students(records: list[dict]) -> int:
    conn = get_connection()
    c = conn.cursor()
    inserted = 0
    for data in records:
        try:
            c.execute("""
                INSERT INTO students
                    (student_identifier, hours_studied, attendance, sleep_hours,
                     previous_score, tutoring_sessions, physical_activity,
                     study_motivation, internet_usage, extracurricular,
                     resources_access, parental_support, target_exam_score)
                VALUES
                    (:student_identifier, :hours_studied, :attendance, :sleep_hours,
                     :previous_score, :tutoring_sessions, :physical_activity,
                     :study_motivation, :internet_usage, :extracurricular,
                     :resources_access, :parental_support, :target_exam_score)
            """, data)
            inserted += 1
        except Exception:
            pass
    conn.commit()
    conn.close()
    return inserted


def get_all_students(page=1, per_page=20, search="", sort_by="id", sort_dir="asc"):
    conn = get_connection()
    c = conn.cursor()

    allowed_cols = {
        "id", "student_identifier", "hours_studied", "attendance",
        "sleep_hours", "previous_score", "tutoring_sessions",
        "target_exam_score", "created_at",
    }
    if sort_by not in allowed_cols:
        sort_by = "id"
    sort_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where = ""
    params_count: list = []
    params_data: list = []
    if search:
        where = "WHERE student_identifier LIKE ?"
        params_count = [f"%{search}%"]
        params_data = [f"%{search}%"]

    c.execute(f"SELECT COUNT(*) FROM students {where}", params_count)
    total = c.fetchone()[0]

    offset = (page - 1) * per_page
    params_data += [per_page, offset]
    c.execute(
        f"SELECT * FROM students {where} ORDER BY {sort_by} {sort_dir} LIMIT ? OFFSET ?",
        params_data,
    )
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows, total


def get_student_by_id(student_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def update_student(student_id: int, data: dict) -> bool:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        UPDATE students SET
            student_identifier=:student_identifier,
            hours_studied=:hours_studied,
            attendance=:attendance,
            sleep_hours=:sleep_hours,
            previous_score=:previous_score,
            tutoring_sessions=:tutoring_sessions,
            physical_activity=:physical_activity,
            study_motivation=:study_motivation,
            internet_usage=:internet_usage,
            extracurricular=:extracurricular,
            resources_access=:resources_access,
            parental_support=:parental_support,
            target_exam_score=:target_exam_score,
            updated_at=datetime('now')
        WHERE id=:id
    """, {**data, "id": student_id})
    conn.commit()
    changed = c.rowcount > 0
    conn.close()
    return changed


def delete_student(student_id: int) -> bool:
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    changed = c.rowcount > 0
    conn.close()
    return changed


def get_student_count() -> int:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM students")
    n = c.fetchone()[0]
    conn.close()
    return n


def get_training_data():
    """Return all students that have a non-null target_exam_score."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM students WHERE target_exam_score IS NOT NULL")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def student_identifier_exists(identifier: str) -> bool:
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT 1 FROM students WHERE student_identifier = ? LIMIT 1",
        (identifier,),
    )
    exists = c.fetchone() is not None
    conn.close()
    return exists


def get_dataset_stats():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT
            COUNT(*) as total,
            AVG(target_exam_score) as avg_score,
            MIN(target_exam_score) as min_score,
            MAX(target_exam_score) as max_score,
            AVG(hours_studied) as avg_hours,
            AVG(attendance) as avg_attendance,
            AVG(sleep_hours) as avg_sleep
        FROM students
        WHERE target_exam_score IS NOT NULL
    """)
    row = c.fetchone()
    conn.close()
    return dict(row) if row else {}


# ─── Predictions ────────────────────────────────────────────

def insert_prediction(data: dict) -> int:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions
            (student_identifier, input_data, predicted_score, performance_level)
        VALUES
            (:student_identifier, :input_data, :predicted_score, :performance_level)
    """, {**data, "input_data": json.dumps(data.get("input_data", {}))})
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id


def get_all_predictions(page=1, per_page=20, search=""):
    conn = get_connection()
    c = conn.cursor()

    where = ""
    params: list = []
    if search:
        where = "WHERE student_identifier LIKE ?"
        params = [f"%{search}%"]

    c.execute(f"SELECT COUNT(*) FROM predictions {where}", params)
    total = c.fetchone()[0]

    offset = (page - 1) * per_page
    c.execute(
        f"SELECT * FROM predictions {where} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [per_page, offset],
    )
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows, total


def delete_prediction(pred_id: int) -> bool:
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    conn.commit()
    changed = c.rowcount > 0
    conn.close()
    return changed


def get_prediction_count() -> int:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM predictions")
    n = c.fetchone()[0]
    conn.close()
    return n


# ─── Model Metrics ──────────────────────────────────────────

def save_model_metrics(data: dict):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO model_metrics
            (model_name, training_records, mae, mse, rmse, r2_score)
        VALUES
            (:model_name, :training_records, :mae, :mse, :rmse, :r2_score)
    """, data)
    conn.commit()
    conn.close()


def get_latest_model_metrics():
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM model_metrics ORDER BY id DESC LIMIT 1"
    )
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None
