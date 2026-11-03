"""
Student Performance Prediction System – Flask Application
"""

import os
import json
import math
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, flash, session,
)
from werkzeug.exceptions import RequestEntityTooLarge

from config import (
    SECRET_KEY, DEBUG, UPLOAD_FOLDER, MAX_CONTENT_LENGTH,
    get_performance_level, ALL_FEATURES,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
import database as db
import data_processor as dp
import ml_model as ml

# ── App setup ────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify(success=False, error="File exceeds 16 MB limit."), 413


# ── Template filters ─────────────────────────────────────────
@app.template_filter("get_level")
def get_level_filter(score):
    if score is None:
        return "Unknown"
    level, _ = get_performance_level(float(score))
    return level.replace(" ", "")

# Also expose as a global for templates
app.jinja_env.globals["get_level"] = get_level_filter


# ── Initialise on startup ────────────────────────────────────
def bootstrap():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    db.init_db()


# ═══════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return redirect(url_for("dashboard"))


# ── Dashboard ────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    stats       = db.get_dataset_stats()
    pred_count  = db.get_prediction_count()
    metrics     = db.get_latest_model_metrics()
    trained     = ml.is_model_trained()
    importances = ml.get_feature_importance() if trained else []

    return render_template(
        "dashboard.html",
        stats=stats,
        pred_count=pred_count,
        metrics=metrics,
        trained=trained,
        importances=importances[:6],
    )


# ── Data management ──────────────────────────────────────────
@app.route("/data")
def data():
    page    = int(request.args.get("page", 1))
    search  = request.args.get("search", "").strip()
    sort_by = request.args.get("sort", "id")
    sort_dir = request.args.get("dir", "asc")
    per_page = 20

    rows, total = db.get_all_students(
        page=page, per_page=per_page,
        search=search, sort_by=sort_by, sort_dir=sort_dir,
    )
    total_pages = max(1, math.ceil(total / per_page))

    return render_template(
        "data.html",
        students=rows,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        search=search,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )


@app.route("/data/add", methods=["GET", "POST"])
def add_data():
    if request.method == "POST":
        errors = []
        f = request.form

        def num(name, lo=None, hi=None, required=True, cast=float):
            val = f.get(name, "").strip()
            if not val:
                if required:
                    errors.append(f"{name} is required.")
                return None
            try:
                v = cast(val)
            except ValueError:
                errors.append(f"{name} must be a number.")
                return None
            if lo is not None and v < lo:
                errors.append(f"{name} must be ≥ {lo}.")
            if hi is not None and v > hi:
                errors.append(f"{name} must be ≤ {hi}.")
            return v

        record = {
            "student_identifier": f.get("student_identifier", "").strip() or None,
            "hours_studied":      num("hours_studied", 0, 24),
            "attendance":         num("attendance", 0, 100),
            "sleep_hours":        num("sleep_hours", 0, 24),
            "previous_score":     num("previous_score", 0, 100),
            "tutoring_sessions":  num("tutoring_sessions", 0, 100, required=False, cast=int),
            "physical_activity":  num("physical_activity", 0, 365, required=False, cast=int),
            "study_motivation":   f.get("study_motivation") or None,
            "internet_usage":     f.get("internet_usage") or None,
            "extracurricular":    f.get("extracurricular") or None,
            "resources_access":   f.get("resources_access") or None,
            "parental_support":   f.get("parental_support") or None,
            "target_exam_score":  num("target_exam_score", 0, 100),
        }

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("add_data.html", form=f)

        db.insert_student(record)
        flash("Student record added successfully.", "success")
        return redirect(url_for("data"))

    return render_template("add_data.html", form={})


@app.route("/data/edit/<int:sid>", methods=["GET", "POST"])
def edit_data(sid):
    student = db.get_student_by_id(sid)
    if not student:
        flash("Record not found.", "error")
        return redirect(url_for("data"))

    if request.method == "POST":
        f = request.form
        updated = {
            "student_identifier": f.get("student_identifier", "").strip() or None,
            "hours_studied":      _safe_float(f.get("hours_studied")),
            "attendance":         _safe_float(f.get("attendance")),
            "sleep_hours":        _safe_float(f.get("sleep_hours")),
            "previous_score":     _safe_float(f.get("previous_score")),
            "tutoring_sessions":  _safe_int(f.get("tutoring_sessions")),
            "physical_activity":  _safe_int(f.get("physical_activity")),
            "study_motivation":   f.get("study_motivation") or None,
            "internet_usage":     f.get("internet_usage") or None,
            "extracurricular":    f.get("extracurricular") or None,
            "resources_access":   f.get("resources_access") or None,
            "parental_support":   f.get("parental_support") or None,
            "target_exam_score":  _safe_float(f.get("target_exam_score")),
        }
        db.update_student(sid, updated)
        flash("Record updated successfully.", "success")
        return redirect(url_for("data"))

    return render_template("add_data.html", form=student, edit=True, edit_id=sid)


@app.route("/data/delete/<int:sid>", methods=["POST"])
def delete_data(sid):
    if db.delete_student(sid):
        flash("Record deleted.", "success")
    else:
        flash("Record not found.", "error")
    return redirect(url_for("data"))


# ── Upload ───────────────────────────────────────────────────
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected.", "error")
            return redirect(request.url)

        if not dp.allowed_file(file.filename):
            flash("Only .csv and .xlsx files are allowed.", "error")
            return redirect(request.url)

        try:
            path = dp.save_upload(file)
            df   = dp.read_file(path)
            df   = dp.map_columns(df)
            df, warnings = dp.clean_dataframe(df)

            # Store preview in session
            preview   = dp.preview_dataframe(df, 10)
            skip_dupes = request.form.get("skip_duplicates", "yes") == "yes"

            # Get existing identifiers for duplicate check
            existing_ids = set()
            if skip_dupes:
                all_rows, _ = db.get_all_students(page=1, per_page=9999999)
                existing_ids = {
                    r["student_identifier"]
                    for r in all_rows
                    if r["student_identifier"]
                }

            records, skipped = dp.df_to_student_dicts(df, skip_dupes, existing_ids)

            # Store in session for confirmation step
            session["pending_upload"] = {
                "records":  records,
                "skipped":  skipped,
                "warnings": warnings,
                "preview":  preview,
                "columns":  list(df.columns),
                "total_raw": len(df) + skipped,
            }
            return render_template(
                "upload_preview.html",
                preview=preview,
                columns=list(df.columns),
                num_records=len(records),
                skipped=skipped,
                warnings=warnings,
                total_raw=len(df) + skipped,
            )

        except Exception as exc:
            flash(f"Error processing file: {exc}", "error")
            return redirect(request.url)

    return render_template("upload.html")


@app.route("/upload/confirm", methods=["POST"])
def upload_confirm():
    pending = session.pop("pending_upload", None)
    if not pending:
        flash("No pending upload. Please upload a file first.", "error")
        return redirect(url_for("upload"))

    records = pending["records"]
    inserted = db.bulk_insert_students(records)
    flash(
        f"Import complete: {inserted} records inserted, "
        f"{pending['skipped']} duplicates skipped.",
        "success",
    )
    return redirect(url_for("data"))


@app.route("/upload/cancel", methods=["POST"])
def upload_cancel():
    session.pop("pending_upload", None)
    flash("Upload cancelled.", "info")
    return redirect(url_for("upload"))


# ── Train ────────────────────────────────────────────────────
@app.route("/train", methods=["GET", "POST"])
def train():
    metrics  = db.get_latest_model_metrics()
    trained  = ml.is_model_trained()
    rows     = db.get_training_data()
    n_rows   = len(rows)

    if request.method == "POST":
        if n_rows < 10:
            flash(
                f"Not enough data to train. Need at least 10 records, have {n_rows}.",
                "error",
            )
            return redirect(url_for("train"))
        try:
            result = ml.train_model(rows)
            db.save_model_metrics({
                "model_name":       "RandomForestRegressor",
                "training_records": result["training_records"],
                "mae":              result["mae"],
                "mse":              result["mse"],
                "rmse":             result["rmse"],
                "r2_score":         result["r2_score"],
            })
            flash(
                f"Model trained successfully on {result['training_records']} records. "
                f"R² = {result['r2_score']:.4f}, RMSE = {result['rmse']:.4f}",
                "success",
            )
            return redirect(url_for("train"))
        except Exception as exc:
            flash(f"Training failed: {exc}", "error")
            return redirect(url_for("train"))

    return render_template(
        "train.html",
        metrics=metrics,
        trained=trained,
        n_rows=n_rows,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )


# ── Predict ──────────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not ml.is_model_trained():
        return render_template("predict.html", not_trained=True)

    if request.method == "POST":
        f = request.form
        try:
            input_data = {
                "Hours_Studied":            float(f.get("hours_studied", 0)),
                "Attendance":               float(f.get("attendance", 0)),
                "Sleep_Hours":              float(f.get("sleep_hours", 0)),
                "Previous_Scores":          float(f.get("previous_score", 0)),
                "Tutoring_Sessions":        int(f.get("tutoring_sessions", 0)),
                "Physical_Activity":        int(f.get("physical_activity", 0)),
                "Motivation_Level":         f.get("study_motivation", "Medium"),
                "Internet_Access":          f.get("internet_usage", "Yes"),
                "Extracurricular_Activities": f.get("extracurricular", "No"),
                "Access_to_Resources":      f.get("resources_access", "Medium"),
                "Parental_Involvement":     f.get("parental_support", "Medium"),
            }

            score = ml.predict(input_data)
            level, color = get_performance_level(score)
            sid   = f.get("student_identifier", "").strip() or "Anonymous"

            db.insert_prediction({
                "student_identifier": sid,
                "input_data":         input_data,
                "predicted_score":    round(score, 2),
                "performance_level":  level,
            })

            return render_template(
                "predict.html",
                result=True,
                score=round(score, 2),
                level=level,
                color=color,
                student_id=sid,
                input_data=input_data,
            )
        except Exception as exc:
            flash(f"Prediction failed: {exc}", "error")
            return redirect(url_for("predict"))

    return render_template("predict.html", not_trained=False)


# ── Prediction history ───────────────────────────────────────
@app.route("/predictions")
def predictions():
    page   = int(request.args.get("page", 1))
    search = request.args.get("search", "").strip()
    per_page = 20

    rows, total = db.get_all_predictions(page=page, per_page=per_page, search=search)
    total_pages = max(1, math.ceil(total / per_page))

    # Parse JSON input_data for display
    for row in rows:
        try:
            row["input_data"] = json.loads(row["input_data"] or "{}")
        except Exception:
            row["input_data"] = {}

    return render_template(
        "predictions.html",
        predictions=rows,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        search=search,
    )


@app.route("/predictions/delete/<int:pid>", methods=["POST"])
def delete_prediction(pid):
    db.delete_prediction(pid)
    flash("Prediction deleted.", "success")
    return redirect(url_for("predictions"))


# ═══════════════════════════════════════════════════════════════
#  JSON API
# ═══════════════════════════════════════════════════════════════

@app.route("/api/model/status")
def api_model_status():
    metrics = db.get_latest_model_metrics()
    return jsonify(
        trained=ml.is_model_trained(),
        metrics=metrics,
        student_count=db.get_student_count(),
        prediction_count=db.get_prediction_count(),
    )


@app.route("/api/students")
def api_students():
    rows, total = db.get_all_students(
        page=int(request.args.get("page", 1)),
        per_page=int(request.args.get("per_page", 20)),
        search=request.args.get("search", ""),
    )
    return jsonify(success=True, data=rows, total=total)


@app.route("/api/students/<int:sid>", methods=["DELETE"])
def api_delete_student(sid):
    if db.delete_student(sid):
        return jsonify(success=True)
    return jsonify(success=False, error="Not found"), 404


@app.route("/api/train", methods=["POST"])
def api_train():
    rows = db.get_training_data()
    try:
        result = ml.train_model(rows)
        db.save_model_metrics({
            "model_name": "RandomForestRegressor",
            **result,
        })
        return jsonify(success=True, metrics=result)
    except Exception as exc:
        return jsonify(success=False, error=str(exc)), 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not ml.is_model_trained():
        return jsonify(success=False, error="Model is not trained yet."), 400
    data = request.get_json(force=True, silent=True) or {}
    try:
        score = ml.predict(data)
        level, color = get_performance_level(score)
        return jsonify(
            success=True,
            predicted_score=round(score, 2),
            performance_level=level,
        )
    except Exception as exc:
        return jsonify(success=False, error=str(exc)), 400


@app.route("/api/dashboard/charts")
def api_dashboard_charts():
    """Return data for Chart.js charts."""
    rows = db.get_training_data()
    scores      = [r["target_exam_score"] for r in rows if r.get("target_exam_score") is not None]
    attendances = [r["attendance"]         for r in rows if r.get("attendance") is not None]
    hours       = [r["hours_studied"]      for r in rows if r.get("hours_studied") is not None]

    def histogram(values, bins=10):
        if not values:
            return [], []
        mn, mx = min(values), max(values)
        if mn == mx:
            return [f"{mn:.0f}"], [len(values)]
        step = (mx - mn) / bins
        labels, counts = [], []
        for i in range(bins):
            lo = mn + i * step
            hi = lo + step
            labels.append(f"{lo:.0f}–{hi:.0f}")
            counts.append(sum(lo <= v < hi for v in values))
        counts[-1] += sum(v == mx for v in values)
        return labels, counts

    sl, sc = histogram(scores)
    al, ac = histogram(attendances)
    hl, hc = histogram(hours)

    # Prediction distribution
    prows, _ = db.get_all_predictions(page=1, per_page=999999)
    p_scores = [r["predicted_score"] for r in prows if r.get("predicted_score") is not None]
    pl, pc = histogram(p_scores)

    return jsonify(
        scores={"labels": sl, "data": sc},
        attendance={"labels": al, "data": ac},
        hours={"labels": hl, "data": hc},
        predictions={"labels": pl, "data": pc},
    )


# ── Helpers ──────────────────────────────────────────────────
def _safe_float(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None

def _safe_int(v):
    f = _safe_float(v)
    return None if f is None else int(f)


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    bootstrap()
    app.run(debug=DEBUG, host="0.0.0.0", port=5000)
