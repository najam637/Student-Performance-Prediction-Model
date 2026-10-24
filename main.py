from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
data = pd.read_csv(BASE_DIR / "student_performance.csv")
FEATURES = [column for column in data.columns if column != "FinalGrade"]
regression_model = joblib.load(BASE_DIR / "regression_model.joblib")
classification_model = joblib.load(BASE_DIR / "classification_model.joblib")
metrics = json.loads((BASE_DIR / "metrics.json").read_text())

app = FastAPI(title="Student Performance Predictor")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

class StudentInput(BaseModel):
    StudyHours: float
    Attendance: float
    Resources: int
    Extracurricular: int
    Motivation: int
    Internet: int
    Gender: int
    Age: int
    LearningStyle: int
    OnlineCourses: int
    Discussions: int
    AssignmentCompletion: int
    ExamScore: float
    EduTech: int
    StressLevel: int

def make_input(student: StudentInput) -> pd.DataFrame:
    values = student.model_dump()
    return pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.get("/dataset/info")
def dataset_info():
    feature_info = {}
    for name in FEATURES:
        values = data[name]
        feature_info[name] = {
            "min": int(values.min()),
            "max": int(values.max()),
            "default": int(values.median())
        }
    return {
        "rows": len(data),
        "features": feature_info,
        "regression": {"r2": metrics["r2"], "mae": metrics["mae"]},
        "classification": {"accuracy": metrics["accuracy"]}
    }

@app.get("/dataset/sample")
def dataset_sample():
    row = data.sample(1, random_state=7).iloc[0]
    return {name: int(row[name]) for name in FEATURES}

@app.post("/predict/regression")
def predict_regression(student: StudentInput):
    row = make_input(student)
    prediction = float(regression_model.predict(row)[0])
    prediction = max(0.0, min(3.0, prediction))
    return {"predicted_final_grade_code": round(prediction, 2)}

@app.post("/predict/classification")
def predict_classification(student: StudentInput):
    row = make_input(student)
    prediction = str(classification_model.predict(row)[0])
    return {"predicted_grade_category": prediction}
