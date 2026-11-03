# 🎓 Student Performance Prediction System

A complete ML-powered web application that predicts student exam performance using a **Random Forest Regressor** trained on academic and lifestyle data.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📤 Dataset Upload | CSV & Excel (.xlsx) with preview & validation |
| ➕ Manual Entry | Add individual student records via form |
| 📋 Data Management | View, search, edit, delete with pagination |
| 🤖 Model Training | Random Forest with MAE / RMSE / R² evaluation |
| 🔮 Prediction | Predict exam score from 11 input features |
| 📜 History | Full prediction history with search |
| 📊 Dashboard | Charts and metrics powered by Chart.js |
| 📱 Responsive | Works on desktop, tablet & mobile |

---

## 🏗 Architecture

```
student-performance-predictor/
├── app.py               # Flask application + routes
├── config.py            # Feature config, performance levels
├── database.py          # SQLite CRUD operations
├── ml_model.py          # RandomForest pipeline: train / predict
├── data_processor.py    # CSV/Excel reading, cleaning, mapping
├── requirements.txt
├── sample_dataset.csv   # 50-row synthetic dataset
├── models/              # Saved model .pkl (created at runtime)
├── uploads/             # Temp upload storage
├── templates/           # Jinja2 HTML templates
└── static/              # CSS + JS
```

**Data Flow (Training)**
```
CSV/Excel/Manual → Flask → Validate → SQLite → RandomForest Pipeline → model.pkl
```

**Data Flow (Prediction)**
```
Student Form → Flask API → Load Pipeline → Predict → Save History → Display Result
```

---

## 🛠 Tech Stack

- **Backend**: Python 3.10+, Flask 3.x
- **ML**: scikit-learn (RandomForestRegressor, Pipeline, ColumnTransformer)
- **Database**: SQLite (via Python sqlite3)
- **Data**: Pandas, NumPy, OpenPyXL
- **Frontend**: HTML5, CSS3, Vanilla JS, Chart.js (CDN)

---

## 🚀 Installation & Running

### 1 · Clone / download the project
```bash
cd student-performance-predictor
```

### 2 · Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3 · Install dependencies
```bash
pip install -r requirements.txt
```

### 4 · Configure environment (optional)
```bash
cp .env.example .env
# Edit .env and set a strong SECRET_KEY for production
```

### 5 · Run the application
```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

The app automatically creates:
- `student_performance.db` (SQLite database)
- `models/` and `uploads/` directories

---

## 📊 Dataset Format

### CSV / Excel columns (recommended)

| Column | Type | Range | Notes |
|---|---|---|---|
| `Hours_Studied` | Float | 0–24 | Daily study hours |
| `Attendance` | Float | 0–100 | Attendance % |
| `Sleep_Hours` | Float | 0–24 | Daily sleep hours |
| `Previous_Scores` | Float | 0–100 | Prior exam score |
| `Tutoring_Sessions` | Integer | 0–100 | Sessions per month |
| `Physical_Activity` | Integer | 0–365 | Active days per year |
| `Motivation_Level` | Text | Low/Medium/High | — |
| `Internet_Access` | Text | Yes/No | — |
| `Extracurricular_Activities` | Text | Yes/No | — |
| `Access_to_Resources` | Text | Low/Medium/High | — |
| `Parental_Involvement` | Text | Low/Medium/High/Very High | — |
| `Exam_Score` | Float | 0–100 | **Target variable** |

Column names are matched **case-insensitively**. Unknown columns are ignored.

---

## 📖 Usage Guide

### Upload Dataset
1. Go to **Upload Dataset**
2. Select a `.csv` or `.xlsx` file
3. Review the 10-row preview and warnings
4. Click **Confirm Import**

### Add Manual Records
1. Go to **Add Student**
2. Fill in the form fields
3. Click **Save Record**

### Train the Model
1. Go to **Train Model**
2. Review the dataset size (need ≥ 10 records)
3. Click **Train Model**
4. Review MAE / RMSE / R² results

### Make a Prediction
1. Go to **Predict**
2. Enter student details
3. Click **Predict Performance**
4. View the predicted score and performance level

---

## 🎯 Performance Levels

| Score | Level |
|---|---|
| 90–100 | Excellent |
| 75–89 | Very Good |
| 60–74 | Good |
| 40–59 | Average |
| 0–39 | Needs Improvement |

---

## 🔧 Configuration

Edit `config.py` to customise:

```python
TARGET_COLUMN = "Exam_Score"          # Change target variable
FEATURE_COLUMNS = [...]               # Change feature list
COLUMN_MAPPING = {                    # Map your CSV columns → DB columns
    "My_Column_Name": "db_column",
    ...
}
RF_PARAMS = {                         # Random Forest hyperparameters
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1,
}
```

---

## 🌐 Deployment

### Render / Railway / Heroku
```bash
# Procfile is already included
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2
```

> **Note**: SQLite is suitable for demos. For a production multi-user app, consider PostgreSQL with SQLAlchemy.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Activate virtual env, run `pip install -r requirements.txt` |
| "Model not trained" | Upload data first, then go to Train Model |
| CSV import fails | Check column names match the expected format |
| Prediction gives weird value | Retrain after adding more diverse data |
| Port 5000 in use | `python app.py` — change port in `app.py` if needed |

---

## ⚠️ Limitations

- Model accuracy depends heavily on dataset quality and size
- SQLite is not suitable for concurrent multi-user production deployments
- No user authentication (add Flask-Login for production)
- Predictions are estimates only — not scientifically validated assessments

---

## 📝 License

MIT — free for educational and personal use.
